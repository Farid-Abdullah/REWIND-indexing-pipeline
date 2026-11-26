"""Run the pipeline on many videos and save everything to disk (no pictures; 3_explore.ipynb shows them).

Main way to run it: 1_pipeline.ipynb (config cell, then Run All). The same functions also run from a terminal:

    python -m rewind.batch --fresh            # delete old results, then every phase on selection.csv
    python -m rewind.batch --stages cpu       # only stages 1-2
    python -m rewind.batch --stages vlm       # only captions (needs phase 1; uses phase 2's frames if they exist)

Phases, in this order, each over every video, so a later problem can't cost you earlier results:
  cpu         stage 1 motion gating -> stage 2 keyframes -> ground-truth checks
  gpu         stage 3 events (YOLO-World + ByteTrack, SigLIP, X-CLIP) -> stage 4 frame selection
              -> index_records.jsonl + index_embeddings.npy
  vlm_clips   stage 5 captions, one per clip (segment cut into <= 60 s): the unit a search returns
  vlm_frames  stage 5 captions, one per selected frame (slow: ~5-10 s per frame on one L4)
Each video is wrapped in try/except: one broken file is logged and skipped, the batch goes on.
cpu / gpu skip finished videos on a re-run (unless force=True); the caption phases resume per clip / frame.

Outputs: outputs/batch/<dataset>/<video>/  (see README)
         outputs/batch/batch_summary.csv    (one row per video, updated after every video)
         outputs/batch/run_log.txt          (everything the run printed, with times)
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from rewind import data as ds
from rewind import keyframes as kfx
from rewind import motion_gating as mg
from rewind.utils.video_utils import format_hms, video_info
from rewind import paths

# Same defaults as the single-video pipeline (motion_gating.SCAN_PARAMS etc.); 1_pipeline.ipynb overrides them.
DEFAULT_CFG = {
    "out_root": paths.BATCH,
    "gate": "mog2",            # gate that feeds keyframes and stages 3-4: "mog2" | "diff" | "fused" | "deep"
    "deep_gate": False,        # DINOv2 gate: a GPU pass over the whole video. Off: docs/pipeline.md found it not worth it
    "scan": {k: v for k, v in mg.SCAN_PARAMS.items()},
    "segments": {
        "gates": ["diff", "mog2"], "threshold_mode": "adaptive", "k": 3.0, "q_noise": 0.5,
        "floor": {"diff": 0.001, "mog2": 0.001, "deep": 0.004},
        "fixed_threshold": {"mog2": 0.004, "diff": 0.006},
        "stop_ratio": 0.5, "smooth_s": 1.0, "merge_gap_s": 3.0, "min_duration_s": 1.0, "pad_s": 2.0,
    },
    "keyframes": {"metric": "phash", "change_threshold": 6.0, "dedup_threshold": 2.5,
                  "min_gap_s": 3.0, "max_gap_s": 30.0, "heartbeat_s": 300.0},
    "v2": {},                  # overrides for gpu_stages.DEFAULT_CONFIG (events, sampling, runtime ...)
    "vlm": {},                 # overrides for captions.CAPTION_CONFIG (model, levels, batch_size, mock ...)
    "naive_fps": 1.0,          # baseline: 1 frame per second
}


def merged_cfg(over: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CFG)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    cfg["out_root"] = Path(cfg["out_root"])
    return cfg


def out_dir(cfg, row) -> Path:
    return cfg["out_root"] / row["dataset"] / row["video_id"]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=_json_default))


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def uniform_times(duration_s: float, n: int) -> np.ndarray:
    """n frames spread evenly over the video: the "same budget, no gating" baseline."""
    return np.linspace(0, duration_s, n + 2)[1:-1] if n > 0 else np.array([])


# ---------------------------------------------------------------------------------------------------
# Phase 1: CPU stages for one video
# ---------------------------------------------------------------------------------------------------
def run_cpu(row, cfg, models=None) -> dict:
    """Stage 1 + 2 + ground-truth checks for one selection row. Writes into out_dir(cfg, row)."""
    video, out = Path(row["path"]), out_dir(cfg, row)
    out.mkdir(parents=True, exist_ok=True)
    T = {}
    info = video_info(video)
    D = info.duration_s
    if D <= 0:
        raise RuntimeError(f"OpenCV reports duration 0 for {video} (unreadable file?)")

    t0 = time.perf_counter()
    scan = mg.scan_video(video, **cfg["scan"], cache_dir=out / "cache")
    T["scan_s"] = time.perf_counter() - t0
    scores, sig, st = scan["scores"], scan["sig"], scan["stats"]

    gates = list(cfg["segments"]["gates"])
    if cfg["deep_gate"] and models is not None:
        from rewind import gpu_stages as gs
        t0 = time.perf_counter()
        v2 = gs.deep_update(gs.DEFAULT_CONFIG, cfg["v2"])
        deep, _ = gs.scan_deep(video, scores, v2, models, cache_dir=out / "cache")
        T["deep_gate_s"] = time.perf_counter() - t0
        if deep is not None:
            scores["score_deep"] = deep
            gates.append("deep")

    seg_kw = cfg["segments"]
    results = mg.segment_methods(scores, D, gates, **seg_kw)
    mg.add_fused(scores, results, D, **seg_kw)
    gate = cfg["gate"] if cfg["gate"] in results else "mog2"
    segments = results[gate]["segments"]
    mg.save_results(out, scores, results, gate)

    t0 = time.perf_counter()
    KF = {m: kfx.select_keyframes(scores, r["segments"], sig, score_col=r["col"], **cfg["keyframes"])
          for m, r in results.items()}
    kf = kfx.export_keyframes(video, KF[gate], out)
    T["keyframes_s"] = time.perf_counter() - t0

    naive = int(D * cfg["naive_fps"])
    s = {"dataset": row["dataset"], "video_id": row["video_id"], "camera_id": row["camera_id"],
         "category": row.get("category", ""), "split": row.get("split", ""),
         "start_wallclock": row.get("start_wallclock", ""), "path": str(video),
         "duration_s": D, "fps": info.fps, "resolution": f"{info.width}x{info.height}",
         "gate": gate, "segments": len(segments),
         "active_pct": 100 * mg.active_fraction(segments, D),
         "threshold_start": results[gate]["start"], "threshold_stop": results[gate]["stop"],
         "keyframes": len(kf), "keyframes_heartbeat": int((kf.segment_id < 0).sum()),
         "naive_frames": naive, "reduction_x": naive / max(1, len(kf)),
         "from_cache": st["from_cache"], **T}
    for m, r in results.items():  # every gate, for comparison in 3_explore.ipynb
        s[f"active_pct_{m}"] = 100 * mg.active_fraction(r["segments"], D)
        s[f"keyframes_{m}"] = len(KF[m])

    # ground truth (MEVA activities / UCA sentences)
    gt = ds.load_gt(row, info.fps, D)
    s["gt_items"] = 0 if gt is None else len(gt)
    if gt is not None and len(gt):
        gt.to_csv(out / "gt.csv", index=False)
        times = scores.timestamp_s.to_numpy()
        g_tab = mg.gate_table(gt, results, times, D)
        sets = {f"keyframes_{m}": KF[m].timestamp_s for m in results}
        sets["uniform_same_count"] = uniform_times(D, len(kf))
        sets["naive_1fps"] = np.arange(0, D, 1 / cfg["naive_fps"])
        k_tab = mg.keyframe_hits(gt, sets)
        g_tab.to_csv(out / "gt_eval_gates.csv")
        k_tab.to_csv(out / "gt_eval_keyframes.csv")
        hit = np.array([bool(np.any((kf.timestamp_s >= a) & (kf.timestamp_s <= b))) for a, b in zip(gt.start_s, gt.end_s)])
        gt.assign(has_keyframe=hit).to_csv(out / "gt.csv", index=False)   # 3_explore.ipynb shows the misses
        s["gt_event_recall"] = float(g_tab.loc[gate, "event_recall"])
        s["gt_frame_recall"] = float(g_tab.loc[gate, "frame_recall"])
        s["gt_frame_precision"] = float(g_tab.loc[gate, "frame_precision"])
        s["gt_kf_hit"] = float(k_tab.loc[f"keyframes_{gate}", "gt_events_with_keyframe"])
        s["gt_kf_hit_uniform"] = float(k_tab.loc["uniform_same_count", "gt_events_with_keyframe"])
        s["gt_coverage_pct"] = 100 * float(mg.active_mask(times, gt).mean())
    s["cpu_s"] = sum(v for k, v in T.items() if k.endswith("_s"))
    s["status_cpu"] = "ok"
    _write_json(out / "summary_cpu.json", s)
    return s


# ---------------------------------------------------------------------------------------------------
# Phase 2: GPU stages for one video (reads phase 1's files)
# ---------------------------------------------------------------------------------------------------
def run_gpu(row, cfg, models, v2) -> dict:
    from rewind import gpu_stages as gs
    video, out = Path(row["path"]), out_dir(cfg, row)
    s1 = _read_json(out / "summary_cpu.json")
    if not s1 or s1.get("status_cpu") != "ok":
        raise RuntimeError("phase 1 results missing; run the CPU stages first")
    ev_out = out / "events"
    ev_out.mkdir(parents=True, exist_ok=True)
    info = video_info(video)
    D = info.duration_s
    scores = pd.read_csv(out / "motion_scores.csv")
    segments = pd.read_csv(out / "segments.csv")
    kf1 = pd.read_csv(out / "keyframes.csv")
    gate, camera_id = s1["gate"], str(row["camera_id"])
    t_start = datetime.fromisoformat(row["start_wallclock"]) if row.get("start_wallclock") else None
    s = {"status_gpu": "ok"}
    if segments.empty:
        s.update(status_gpu="no segments", gpu_s=0.0)
        _write_json(out / "summary_gpu.json", s)
        return s

    t0 = time.perf_counter()
    feat = gs.extract_event_features(video, info, segments, v2, models)
    meta, E = feat["meta"], feat["emb"]
    use = v2["events"]["methods"]
    ev_tr, track_sum = gs.track_events(feat["tracks"], v2) if "tracks" in use and len(feat["tracks"]) \
        else (pd.DataFrame(), pd.DataFrame())
    ev_sg, sg_extra = gs.siglip_events(feat, v2, models) if "siglip" in use else (pd.DataFrame(), None)
    ev_xc = gs.xclip_events(feat, v2) if "xclip" in use else pd.DataFrame()
    events = gs.merge_events([ev_tr, ev_sg, ev_xc], v2["events"]["merge_gap_s"])
    scores["score_ratio"] = scores[f"score_{gate}"] / s1["threshold_start"]
    imp = gs.importance_curve(meta, scores, "score_ratio", events, track_sum, v2)
    anchors = gs.event_anchors(meta, events, track_sum, v2)
    meta["importance"] = imp
    if sg_extra is not None:
        meta["siglip_tag"] = sg_extra["tags"]
    if t_start is not None and len(events):
        events["start_wallclock"] = [(t_start + timedelta(seconds=float(x))).isoformat() for x in events.start_s]
    if len(events):
        events.insert(1, "camera_id", camera_id)
    events.to_csv(ev_out / "events.csv", index=False)
    feat["tracks"].to_csv(ev_out / "tracks_raw.csv", index=False)
    track_sum.to_csv(ev_out / "tracks.csv", index=False)
    feat["xclip"].to_csv(ev_out / "xclip_windows.csv", index=False)
    meta.drop(columns="phash").to_csv(ev_out / "candidate_frames.csv", index=False)
    s["stage3_s"] = time.perf_counter() - t0
    s["stage3_frames"] = feat["stats"]["frames"]

    # stage 4: samplers, scored side by side (v1 keyframes moved to the nearest candidate frame)
    t0 = time.perf_counter()
    c = v2["sampling"]
    keeps = {}
    for name in c["methods"]:
        if name != "phash" and E.shape[1] == 0:
            continue
        keeps[name] = gs.SAMPLERS[name](meta, c, E=E, imp=imp, anchors=anchors)
    keeps["v1_keyframes"] = _nearest_candidates(meta, kf1)
    evals = {n: gs.evaluate_sampling(meta, E, k, events, track_sum, info, c) for n, k in keeps.items()}
    pd.DataFrame(evals).T.to_csv(ev_out / "sampling_comparison.csv")
    psamp = c["primary"] if c["primary"] in keeps else next(iter(keeps))
    mt = meta.timestamp_s.to_numpy()
    for name, keep in keeps.items():
        kf_s = meta.loc[keep].assign(camera_id=camera_id, sampler=name)
        kf_s["timestamp_hms"] = kf_s.timestamp_s.map(format_hms)
        if name in c["export_jpgs"]:
            kf_s["filename"] = gs.export_keyframes(video, info, meta, keep, ev_out, name, camera_id)
        kf_s.drop(columns="phash").to_csv(ev_out / f"keyframes_{name}.csv")
    rec_path = gs.write_index_records(ev_out, meta, keeps[psamp], E, events, track_sum, camera_id,
                                      video.name, t_start)
    gt_path = out / "gt.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path)
        if len(gt):
            mg.keyframe_hits(gt, {n: mt[np.array(k, int)] for n, k in keeps.items()}).to_csv(ev_out / "gt_eval_sampling.csv")
    s["stage4_s"] = time.perf_counter() - t0
    try:  # a picture for 3_explore.ipynb; never fail the batch over it
        gate_res = {gate: {"col": f"score_{gate}", "start": s1["threshold_start"], "segments": segments}}
        gs.plot_timeline(ev_out / "timeline.png", scores, gate_res, gate, events,
                         {n: mt[np.array(k, int)] for n, k in keeps.items()}, info)
    except Exception:
        pass
    s.update({"events": len(events), "events_confirmed": int((events.support >= 2).sum()) if len(events) else 0,
              "tracks": len(track_sum), "candidate_frames": len(meta), "primary_sampler": psamp,
              "index_records": sum(1 for _ in open(rec_path)), "embedding_dim": int(E.shape[1]),
              "skipped_models": "; ".join(f"{k}: {v[:80]}" for k, v in models.failed.items()),
              "gpu_s": s["stage3_s"] + s["stage4_s"]})
    _write_json(out / "summary_gpu.json", s)
    return s


# ---------------------------------------------------------------------------------------------------
# Phase 3: VLM captions for one video (reads phase 1's files, and phase 2's if they exist)
# ---------------------------------------------------------------------------------------------------
VLM_PHASES = {"vlm_clips": ["clips"], "vlm_frames": ["frames"]}   # "vlm" = both, clips first


def run_vlm(row, cfg, captioner, levels, force=False) -> dict:
    from rewind import captions as cap
    return cap.caption_video(out_dir(cfg, row), captioner, force=force, levels=levels)


def vlm_devices(cfg) -> list:
    """CFG["vlm"]["device"] as a list: "cuda:0" -> ["cuda:0"], ["cuda:0", "cuda:1"] stays."""
    from rewind import captions as cap
    dev = {**cap.CAPTION_CONFIG, **cfg.get("vlm", {})}["device"]
    return list(dev) if isinstance(dev, (list, tuple)) else [dev]


def split_by_duration(rows: list, k: int) -> list[list]:
    """Videos into k groups of about equal total duration (longest first, each to the least-loaded group)."""
    groups, load = [[] for _ in range(k)], [0.0] * k
    for r in sorted(rows, key=lambda r: -float(r["duration_s"] or 0)):
        j = load.index(min(load))
        groups[j].append(r)
        load[j] += float(r["duration_s"] or 0)
    return groups


def caption_one(row, cfg, captioner, phase, force, tag, log) -> None:
    """One video, one caption level; logs one line, writes error_<phase>.json on failure."""
    out = out_dir(cfg, row)
    t0 = time.perf_counter()
    try:
        s = run_vlm(row, cfg, captioner, VLM_PHASES[phase], force)
        level = VLM_PHASES[phase][0]
        new, have = s[f"vlm_{level}_new"], s[f"vlm_{level}"]
        msg = (f"{have} {level} captioned ({new} new, from {s['vlm_source']}), {s['vlm_parse_failed']} unparsed"
               if new or have else "nothing to caption")
        (out / f"error_{phase}.json").unlink(missing_ok=True)
        log(f"{tag}: {msg}  [{time.perf_counter() - t0:.0f} s]")
    except Exception as e:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / f"error_{phase}.json", {f"status_{phase}": f"error: {e!r}"[:300],
                                                  f"traceback_{phase}": traceback.format_exc()[-2000:]})
        log(f"{tag}: ERROR {e!r}  (details in {out / f'error_{phase}.json'})")


def _vlm_worker(name, device, jobs, cfg, phase, force, queue) -> None:
    """Runs in its own process: one VLM on `device`, captions its share of the videos, sends log lines back."""
    try:
        from rewind import captions as cap

        def log(msg):
            queue.put(("log", f"[{name}] {msg}"))
        cap.log = lambda m: log(f"[captions] {m}")   # caption_video's progress lines go to the notebook too
        captioner = cap.Captioner({**cfg["vlm"], "device": device})
        for tag, row in jobs:
            caption_one(row, cfg, captioner, phase, force, tag, log)
            queue.put(("video_done", name))
    except Exception as e:  # report, don't hang the notebook
        queue.put(("log", f"[{name}] WORKER ERROR {e!r}\n{traceback.format_exc()[-1500:]}"))
    finally:
        queue.put(("worker_done", name))


def caption_parallel(selection, cfg, phase, force, devices, log) -> None:
    """One worker process per GPU, each with its own copy of the VLM and about half (1/k) of the video-time."""
    import multiprocessing as mp
    import queue as queue_mod
    n = len(selection)
    rows = [dict(r, _tag=f"[{i}/{n}] {r['dataset']}/{r['video_id']}")
            for i, r in enumerate(selection.to_dict("records"), 1)]
    groups = split_by_duration(rows, len(devices))
    ctx = mp.get_context("spawn")   # CUDA can't be forked; spawn works from Jupyter too
    q = ctx.Queue()
    procs = []
    names = [d if devices.count(d) == 1 else f"{d}#{i}" for i, d in enumerate(devices)]
    for name, dev, grp in zip(names, devices, groups):
        h = sum(float(r["duration_s"] or 0) for r in grp) / 3600
        log(f"{name}: {len(grp)} videos, {h:.2f} h (loading its own copy of the VLM)")
        jobs = [(r.pop("_tag"), r) for r in grp]
        p = ctx.Process(target=_vlm_worker, args=(name, dev, jobs, cfg, phase, force, q), daemon=True)
        p.start()
        procs.append((name, p))
    running = {name for name, _ in procs}
    while running:
        try:
            kind, payload = q.get(timeout=5)
        except queue_mod.Empty:
            for dev, p in procs:   # a worker killed from outside (e.g. out of RAM) sends nothing
                if dev in running and not p.is_alive():
                    log(f"[{dev}] worker stopped unexpectedly (exit code {p.exitcode}); its remaining videos "
                        f"were not captioned. Re-run the cell (FRESH_RUN = False) to continue.")
                    running.discard(dev)
            continue
        if kind == "log":
            log(payload)
        elif kind == "video_done":
            collect_summary(cfg, selection)
        elif kind == "worker_done":
            running.discard(payload)
    for _, p in procs:
        p.join(timeout=30)


def free_gpu_memory(models=None) -> None:
    """Drop the phase-2 models (if any) and return cached GPU memory before the VLM loads (~16 GB of 24)."""
    if models is not None:
        models._cache.clear()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _nearest_candidates(meta, kf):
    """Candidate-row index nearest to each v1 keyframe, within the same segment (heartbeats: idle rows)."""
    t, sid = meta.timestamp_s.to_numpy(), meta.segment_id.to_numpy()
    out = set()
    for r in kf.itertuples():
        m_ = np.flatnonzero(sid == r.segment_id) if r.segment_id >= 0 else np.flatnonzero(sid < 0)
        if m_.size:
            out.add(int(m_[np.argmin(np.abs(t[m_] - r.timestamp_s))]))
    return sorted(out)


# ---------------------------------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------------------------------
PHASES = ("cpu", "gpu", "vlm_clips", "vlm_frames")
SUMMARY_FILES = ("summary_cpu.json", "summary_gpu.json", "summary_vlm.json",
                 "error_cpu.json", "error_gpu.json", "error_vlm_clips.json", "error_vlm_frames.json")


def expand_stages(stages) -> list[str]:
    """"vlm" -> both caption passes; keeps the fixed order cpu, gpu, vlm_clips, vlm_frames."""
    want = set()
    for st in stages:
        want |= set(VLM_PHASES) if st == "vlm" else {st}
    bad = want - set(PHASES)
    if bad:
        raise ValueError(f"unknown stage(s) {sorted(bad)}; use {list(PHASES) + ['vlm']}")
    return [p for p in PHASES if p in want]


def clean_outputs(out_root, keep_decode_cache: bool = False, log=print) -> None:
    """Fresh start: delete the batch's results under out_root (every <dataset>/<video>/ folder and the
    batch-level files). keep_decode_cache=True keeps each video's cache/ (stage 1's slow decode pass;
    it is keyed by the scan settings, so it can't give stale results, but stage 1 times then read ~0)."""
    import shutil
    out_root = Path(out_root).resolve()
    if not out_root.exists():
        return
    if out_root == Path(out_root.anchor) or out_root == Path.home().resolve() or out_root == paths.REPO:
        raise ValueError(f"refusing to clean {out_root}")
    n = 0
    for child in out_root.iterdir():
        if child.is_dir():
            for vd in child.iterdir():   # <dataset>/<video>
                if not vd.is_dir():
                    vd.unlink()
                    continue
                for item in vd.iterdir():
                    if keep_decode_cache and item.name == "cache":
                        continue
                    shutil.rmtree(item) if item.is_dir() else item.unlink()
                if not any(vd.iterdir()):
                    vd.rmdir()
                n += 1
            if not any(child.iterdir()):
                child.rmdir()
        else:
            child.unlink()
    log(f"fresh start: cleared {n} video folders in {out_root}" + (" (decode cache kept)" if keep_decode_cache else ""))


def make_logger(path):
    """print() + append to a log file with the time, so a run can be followed (`tail -f`) or read after the
    notebook lost its connection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(path, "a") as f:
            for line in str(msg).splitlines() or [""]:
                f.write(f"{datetime.now():%H:%M:%S} {line}\n")
    return log


def collect_summary(cfg, selection: pd.DataFrame) -> pd.DataFrame:
    """One row per selected video from the summary_*.json files on disk (also works for a half-done batch)."""
    rows = []
    for _, row in selection.iterrows():
        out = out_dir(cfg, row)
        r = {"dataset": row["dataset"], "video_id": row["video_id"]}
        for f in SUMMARY_FILES:
            r.update(_read_json(out / f) or {})
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(cfg["out_root"] / "batch_summary.csv", index=False)
    return df


def run_batch(selection: pd.DataFrame, cfg: dict | None = None, stages=("cpu", "gpu", "vlm"), force=False,
              log=print, captioner=None) -> pd.DataFrame:
    """Run the given phases over every selected video, in the order cpu, gpu, vlm_clips, vlm_frames.
    cpu / gpu skip videos that finished before (unless force). The caption phases always look at every
    video, but caption only what is missing (per frame / per clip), so they resume where they stopped.
    Pass `captioner` (captions.Captioner) to reuse a loaded VLM between calls. If CFG["vlm"]["device"] is a list
    of GPUs, each caption phase runs one worker process per GPU instead (each loads its own VLM; `captioner` unused)."""
    cfg = merged_cfg(cfg)
    phases = expand_stages(stages)
    cfg["out_root"].mkdir(parents=True, exist_ok=True)
    selection.to_csv(cfg["out_root"] / "selection_used.csv", index=False)
    old = _read_json(cfg["out_root"] / "batch_config.json") or {}
    ran = sorted(set(old.get("stages", [])) | set(phases), key=lambda p: PHASES.index(p) if p in PHASES else 99)
    _write_json(cfg["out_root"] / "batch_config.json", {**cfg, "stages": ran})   # 3_explore.ipynb reads it
    n = len(selection)
    total_h = selection.duration_s.astype(float).sum() / 3600
    log(f"{n} videos, {total_h:.2f} h of video -> {cfg['out_root']}  (phases: {', '.join(phases)})")

    models = v2 = None
    if "gpu" in phases or ("cpu" in phases and cfg["deep_gate"]):
        from rewind import gpu_stages as gs
        v2 = gs.deep_update(gs.DEFAULT_CONFIG, cfg["v2"])
        models = gs.Models(v2)
        log(f"GPU stages: device {models.device}" + ("  (MOCK MODELS)" if models.mock else ""))

    devices = vlm_devices(cfg)
    for phase in phases:
        log(f"\n=== phase {phase.upper()} ===")
        is_vlm = phase in VLM_PHASES
        if is_vlm and len(devices) > 1:   # one VLM per GPU, in worker processes
            free_gpu_memory(models)
            models = None
            t_phase = time.perf_counter()
            log(f"captioning {VLM_PHASES[phase][0]} on {len(devices)} GPUs: {', '.join(devices)}")
            caption_parallel(selection, cfg, phase, force, devices, log)
            log(f"phase {phase.upper()} took {(time.perf_counter() - t_phase) / 60:.1f} min")
            collect_summary(cfg, selection)
            continue
        if is_vlm and captioner is None:
            from rewind import captions as cap
            free_gpu_memory(models)
            models = None
            captioner = cap.Captioner(cfg["vlm"])
        if is_vlm:
            log(f"VLM: {captioner.model_name}" + ("  (MOCK)" if captioner.cfg["mock"] else "") +
                f", captioning {VLM_PHASES[phase][0]}")
        t_phase = time.perf_counter()
        for i, (_, row) in enumerate(selection.iterrows(), 1):
            out = out_dir(cfg, row)
            tag = f"[{i}/{n}] {row['dataset']}/{row['video_id']}"
            if not is_vlm:
                done = _read_json(out / f"summary_{phase}.json")
                if done and str(done.get(f"status_{phase}", "")).startswith(("ok", "no segments")) and not force:
                    log(f"{tag}: done before, skipped")
                    continue
            if is_vlm:
                caption_one(row, cfg, captioner, phase, force, tag, log)
                collect_summary(cfg, selection)
                continue
            t0 = time.perf_counter()
            try:
                if phase == "cpu":
                    s = run_cpu(row, cfg, models)
                    msg = (f"{s['duration_s'] / 60:.1f} min, {s['active_pct']:.0f}% active, {s['keyframes']} kf "
                           f"({s['reduction_x']:.0f}x fewer)" +
                           (f", GT {s['gt_items']} items, {100 * s['gt_kf_hit']:.0f}% with kf" if s.get("gt_kf_hit") is not None
                            and s.get("gt_items") else ", no GT"))
                else:
                    s = run_gpu(row, cfg, models, v2)
                    msg = s["status_gpu"] if s["status_gpu"] != "ok" else \
                        f"{s['events']} events, {s['tracks']} tracks, {s['index_records']} index records"
                (out / f"error_{phase}.json").unlink(missing_ok=True)
                log(f"{tag}: {msg}  [{time.perf_counter() - t0:.0f} s]")
            except Exception as e:
                out.mkdir(parents=True, exist_ok=True)
                _write_json(out / f"error_{phase}.json", {f"status_{phase}": f"error: {e!r}"[:300],
                                                          f"traceback_{phase}": traceback.format_exc()[-2000:]})
                log(f"{tag}: ERROR {e!r}  (details in {out / f'error_{phase}.json'})")
            collect_summary(cfg, selection)
        log(f"phase {phase.upper()} took {(time.perf_counter() - t_phase) / 60:.1f} min")
    return collect_summary(cfg, selection)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the pipeline on every video in selection.csv "
                                             "(same as 1_pipeline.ipynb, which is the main way to run it).")
    ap.add_argument("--selection", default=str(paths.SELECTION))
    ap.add_argument("--data-root", default=str(ds.DATA_ROOT))
    ap.add_argument("--stages", nargs="+", default=["cpu", "gpu", "vlm"], choices=list(PHASES) + ["vlm"])
    ap.add_argument("--meva-min", type=float, default=60)
    ap.add_argument("--uca-min", type=float, default=60)
    ap.add_argument("--fresh", action="store_true", help="delete all results in outputs/batch first")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mock-models", action="store_true", help="fake GPU models + captioner: plumbing check only")
    ap.add_argument("--vlm-model", help="captioning model: HF name or local folder (default: captions.CAPTION_CONFIG)")
    a = ap.parse_args()
    sel = ds.load_or_make_selection(a.selection, Path(a.data_root), meva_min=a.meva_min, uca_min=a.uca_min)
    cfg = {"v2": {"runtime": {"mock_models": True}}, "vlm": {"mock": True}} if a.mock_models else {"vlm": {}}
    if a.vlm_model:
        cfg["vlm"]["model"] = a.vlm_model
    log = make_logger(DEFAULT_CFG["out_root"] / "run_log.txt")
    if a.fresh:
        clean_outputs(DEFAULT_CFG["out_root"], log=log)
    summ = run_batch(sel, cfg, stages=a.stages, force=a.force, log=log)
    print(f"\nsummary: {DEFAULT_CFG['out_root'] / 'batch_summary.csv'} ({len(summ)} rows)")


if __name__ == "__main__":
    main()
