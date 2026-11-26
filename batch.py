"""Run the pipeline on many videos and save everything to disk (no pictures; explore.ipynb shows them).

Used by pipeline.ipynb (config cell + Run All). Also runs from a terminal, e.g. inside tmux so it survives
a closed laptop:

    python batch.py                    # selection.csv (made on first run), CPU stages then GPU stages
    python batch.py --stages cpu       # only stages 1-2
    python batch.py --stages vlm       # only captions (needs phase 1; uses phase 2's frames if they exist)
    python batch.py --force            # redo videos that already have results

Three phases, so a GPU problem can't cost you the CPU results:
  phase 1 (CPU, every video): stage 1 motion gating -> stage 2 keyframes -> ground-truth checks
  phase 2 (GPU, every video): stage 3 events (YOLO-World + ByteTrack, SigLIP, X-CLIP) -> stage 4 frame
                               selection -> index_records.jsonl + index_embeddings.npy
  phase 3 (VLM, every video): stage 5 captions (captions.py): one per selected frame + one per clip
                               -> captions/captions.csv; then caption_review.csv for a hand check
Each video is wrapped in try/except: one broken file is logged and skipped, the batch goes on.
Finished videos are skipped on a re-run (resume), unless force=True.

Outputs: outputs/batch/<dataset>/<video>/  (same files as the single-video notebook, see README)
         outputs/batch/batch_summary.csv    (one row per video, updated after every video)
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

import datasets as ds
import keyframes as kfx
import motion_gating as mg
from video_utils import format_hms, video_info

ROOT = Path(__file__).resolve().parent

# Same defaults as the single-video pipeline (motion_gating.SCAN_PARAMS etc.); pipeline.ipynb overrides them.
DEFAULT_CFG = {
    "out_root": ROOT / "outputs" / "batch",
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
        import gpu_stages as gs
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
    for m, r in results.items():  # every gate, for comparison in explore.ipynb
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
        gt.assign(has_keyframe=hit).to_csv(out / "gt.csv", index=False)   # explore.ipynb shows the misses
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
    import gpu_stages as gs
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
    try:  # a picture for explore.ipynb; never fail the batch over it
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
def run_vlm(row, cfg, captioner, force=False) -> dict:
    import captions as cap
    return cap.caption_video(out_dir(cfg, row), captioner, force=force)


def _free_gpu_models(models) -> None:
    """Drop the phase-2 models before the VLM loads (it needs ~16 GB of the L4's 24 GB)."""
    if models is None:
        return
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
def collect_summary(cfg, selection: pd.DataFrame) -> pd.DataFrame:
    """One row per selected video from the summary_*.json files on disk (also works for a half-done batch)."""
    rows = []
    for _, row in selection.iterrows():
        out = out_dir(cfg, row)
        r = {"dataset": row["dataset"], "video_id": row["video_id"]}
        for f in ("summary_cpu.json", "summary_gpu.json", "summary_vlm.json",
                  "error_cpu.json", "error_gpu.json", "error_vlm.json"):
            r.update(_read_json(out / f) or {})
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(cfg["out_root"] / "batch_summary.csv", index=False)
    return df


def run_batch(selection: pd.DataFrame, cfg: dict | None = None, stages=("cpu", "gpu", "vlm"), force=False,
              log=print) -> pd.DataFrame:
    cfg = merged_cfg(cfg)
    cfg["out_root"].mkdir(parents=True, exist_ok=True)
    selection.to_csv(cfg["out_root"] / "selection_used.csv", index=False)
    _write_json(cfg["out_root"] / "batch_config.json", {**cfg, "stages": list(stages)})   # explore.ipynb reads it
    n = len(selection)
    total_h = selection.duration_s.astype(float).sum() / 3600
    log(f"{n} videos, {total_h:.2f} h of video -> {cfg['out_root']}")

    models = v2 = None
    if "gpu" in stages or cfg["deep_gate"]:
        import gpu_stages as gs
        v2 = gs.deep_update(gs.DEFAULT_CONFIG, cfg["v2"])
        models = gs.Models(v2)
        log(f"GPU stages: device {models.device}" + ("  (MOCK MODELS)" if models.mock else ""))

    captioner = None
    for phase in [p for p in ("cpu", "gpu", "vlm") if p in stages]:
        log(f"\n=== phase {phase.upper()} ===")
        if phase == "vlm":
            import captions as cap
            _free_gpu_models(models)
            captioner = cap.Captioner(cfg["vlm"])
            log(f"VLM: {captioner.model_name}" + ("  (MOCK)" if captioner.cfg["mock"] else ""))
        t_phase = time.perf_counter()
        for i, (_, row) in enumerate(selection.iterrows(), 1):
            out = out_dir(cfg, row)
            done = _read_json(out / f"summary_{phase}.json")
            tag = f"[{i}/{n}] {row['dataset']}/{row['video_id']}"
            if phase == "vlm" and done and done.get("settings_hash") != captioner.settings_hash():
                done = None   # other model / prompts / image size: caption again
            if done and str(done.get(f"status_{phase}", "")).startswith(("ok", "no segments")) and not force:
                log(f"{tag}: done before, skipped")
                continue
            t0 = time.perf_counter()
            try:
                if phase == "vlm":
                    s = run_vlm(row, cfg, captioner, force)
                    msg = (f"{s['vlm_frames']} frame + {s['vlm_clips']} clip captions ({s['vlm_source']}), "
                           f"{s['vlm_parse_failed']} unparsed")
                elif phase == "cpu":
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
        if phase == "vlm":
            sheet = cap.make_review_sheet(cfg["out_root"], cfg.get("review_n", 60))
            if sheet:
                log(f"caption hand-check sheet: {sheet} (+ caption_review.html)")
    return collect_summary(cfg, selection)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the pipeline on every video in selection.csv.")
    ap.add_argument("--selection", default=str(ROOT / "selection.csv"))
    ap.add_argument("--data-root", default=str(ds.DATA_ROOT))
    ap.add_argument("--stages", nargs="+", default=["cpu", "gpu", "vlm"], choices=["cpu", "gpu", "vlm"])
    ap.add_argument("--meva-min", type=float, default=60)
    ap.add_argument("--uca-min", type=float, default=60)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mock-models", action="store_true", help="fake GPU models + captioner: plumbing check only")
    ap.add_argument("--vlm-model", help="captioning model: HF name or local folder (default: captions.CAPTION_CONFIG)")
    a = ap.parse_args()
    sel = ds.load_or_make_selection(a.selection, Path(a.data_root), meva_min=a.meva_min, uca_min=a.uca_min)
    cfg = {"v2": {"runtime": {"mock_models": True}}, "vlm": {"mock": True}} if a.mock_models else {"vlm": {}}
    if a.vlm_model:
        cfg["vlm"]["model"] = a.vlm_model
    summ = run_batch(sel, cfg, stages=a.stages, force=a.force, log=lambda m: print(m, flush=True))
    print(f"\nsummary: {DEFAULT_CFG['out_root'] / 'batch_summary.csv'} ({len(summ)} rows)")


if __name__ == "__main__":
    main()
