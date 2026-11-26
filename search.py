"""Search: type a sentence, get ranked clips with times. And the evaluation that scores it against ground truth.

Used by search.ipynb (build, evaluate, try a query) and explore.ipynb section 8 (look at the results).
Reads what pipeline.ipynb saved in outputs/batch/; writes outputs/batch/search/.

Methods compared (all return moments: video + start + end):
  captions            the query vs each clip's caption text (sentence embeddings, bge-small)          -> clips
  siglip_selected     the query vs the SigLIP embeddings of the frames stage 4 selected               -> frames
  hybrid              both rankings merged by reciprocal rank fusion (a clip's SigLIP score = its best frame)
  siglip_1fps         baseline: SigLIP on frames sampled evenly at 1 per second, no gating, no captions
  clip_1fps           baseline: the same with CLIP
Each result has two spans: the whole clip ("clip") and a short window around its best-matching frame
("window", `moment_s` long), because a 60-s clip can't reach temporal IoU 0.3 on a 5-s event.

Queries (queries.csv, made from the ground truth on the first run, editable, add your own rows):
  uca   every UCA sentence of the selected UCA videos; its answer = its own start-end in its video
  meva  one sentence per MEVA activity type (e.g. "a person opens a door"); answers = every labelled instance

Scores per method, dataset and setting ("corpus" = search all videos of the dataset; "video" = only the
query's own video, UCA only, as in the UCA paper): Recall@1/5/10 at temporal IoU 0.3 and 0.5, a lenient
Recall@K (a returned span overlaps an answer at all), MRR at IoU 0.3, and how many queries the gating left
reachable at all (an answer overlaps some clip), so a miss can be traced to gating or to search.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import caption_eval as ce
from video_utils import read_frames, video_info

SEARCH_CONFIG = {
    "text_model": ce.TEXT_MODEL,               # sentence embeddings for captions and queries
    "query_prefix": "Represent this sentence for searching relevant passages: ",  # bge's retrieval prefix (queries only)
    "siglip_model": "google/siglip-base-patch16-224",   # must be the stage-3 model: the hand-off embeddings are its
    "clip_model": "openai/clip-vit-base-patch32",        # baseline only
    "baselines": True,                         # embed 1 frame/s of every video with SigLIP and CLIP (a few minutes)
    "baseline_fps": 1.0,
    "rrf_k": 60,                               # reciprocal rank fusion constant
    "top_k": 10,
    "moment_s": 10.0,                          # "window" span: this many seconds around the best frame
    "nms_s": 10.0,                             # frame methods: frames closer than this in one video = one result
    "iou": [0.3, 0.5],
    "ks": [1, 5, 10],
    "device": "auto",
    "batch": 64,
    "mock": False,                             # fake encoders: plumbing check only
}
METHODS = ["captions", "siglip_selected", "hybrid", "siglip_1fps", "clip_1fps"]
FRAME_METHODS = {"siglip_selected", "siglip_1fps", "clip_1fps"}

# MEVA activity names -> a sentence an operator might type
MEVA_QUERIES = {
    "person_talks_to_person": "two people talking to each other",
    "person_exits_scene_through_structure": "a person leaves through a door",
    "person_enters_scene_through_structure": "a person comes in through a door",
    "person_picks_up_object": "a person picks up an object",
    "person_puts_down_object": "a person puts an object down",
    "person_texts_on_phone": "a person texting on a phone",
    "person_talks_on_phone": "a person talking on a phone",
    "person_transfers_object": "a person hands an object to another person",
    "hand_interacts_with_person": "a person touches another person with their hand",
    "person_opens_facility_door": "a person opens a door",
    "person_closes_facility_door": "a person closes a door",
    "person_stands_up": "a person stands up",
    "person_sits_down": "a person sits down",
    "person_carries_heavy_object": "a person carrying a heavy object",
    "person_embraces_person": "two people hugging",
    "person_opens_vehicle_door": "a person opens a car door",
    "person_closes_vehicle_door": "a person closes a car door",
    "person_enters_vehicle": "a person gets into a car",
    "person_exits_vehicle": "a person gets out of a car",
    "person_loads_vehicle": "a person loads something into a car",
    "person_unloads_vehicle": "a person unloads something from a car",
    "person_opens_trunk": "a person opens the trunk of a car",
    "person_closes_trunk": "a person closes the trunk of a car",
    "person_rides_bicycle": "a person riding a bicycle",
    "person_abandons_package": "a person leaves a package behind",
    "person_purchases": "a person buying something at a counter",
    "person_reads_document": "a person reading a document",
    "person_interacts_with_laptop": "a person using a laptop",
    "person_steals_object": "a person stealing an object",
    "vehicle_turns_left": "a car turning left",
    "vehicle_turns_right": "a car turning right",
    "vehicle_starts": "a parked car starts moving",
    "vehicle_stops": "a car stops",
    "vehicle_reverses": "a car reversing",
    "vehicle_makes_u_turn": "a car making a U-turn",
    "vehicle_picks_up_person": "a car picks up a person",
    "vehicle_drops_off_person": "a car drops off a person",
}


def log(msg):
    print(f"[search] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------
# Encoders (lazy; mock versions for plumbing checks)
# ---------------------------------------------------------------------------------------------------
class _Mock:
    """Fake image/text encoder (no model): deterministic vectors, for plumbing checks only."""

    def __init__(self, dim=64, seed=0):
        import gpu_stages as gs
        self.m = gs.MockSigLIP(dim)
        self.seed = seed

    def images(self, frames_bgr):
        return self.m.images(frames_bgr)

    def texts(self, texts):
        return self.m.texts([f"{self.seed}:{t}" if self.seed else t for t in texts])


class _HFImageText:
    """SigLIP or CLIP through transformers: unit-length image and text embeddings in one space."""

    def __init__(self, name, device, siglip):
        import torch
        from transformers import AutoModel, AutoProcessor
        from gpu_stages import _as_tensor
        self.torch, self.device, self.siglip, self._as_tensor = torch, device, siglip, _as_tensor
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(name).to(device, self.dtype).eval()
        self.proc = AutoProcessor.from_pretrained(name)

    def images(self, frames_bgr):
        import cv2
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        with self.torch.inference_mode():
            px = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            e = self._as_tensor(self.model.get_image_features(pixel_values=px)).float()
            return self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    def texts(self, texts):
        with self.torch.inference_mode():
            if self.siglip:   # SigLIP was trained with max_length padding
                tok = self.proc(text=texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
                e = self.model.get_text_features(input_ids=tok["input_ids"].to(self.device))
            else:
                tok = self.proc(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                e = self.model.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            e = self._as_tensor(e).float()
            return self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def _device(d):
    if d != "auto":
        return d
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------------------------------
# Spans and scoring
# ---------------------------------------------------------------------------------------------------
def iou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def window(t, length, duration):
    """`length` seconds centred on t, kept inside [0, duration]."""
    s = max(0.0, t - length / 2)
    e = min(duration, s + length) if duration else s + length
    return max(0.0, e - length), e


# ---------------------------------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------------------------------
class SearchIndex:
    """Everything a query is compared against: clips (captions + caption embeddings), the selected frames
    (SigLIP embeddings from stage 4), and for the baselines 1 frame/s of every video (SigLIP + CLIP).
    Built from outputs/batch/ and cached in outputs/batch/search/index/."""

    def __init__(self, out_root, cfg=None):
        self.out_root = Path(out_root)
        self.cfg = {**SEARCH_CONFIG, **(cfg or {})}
        self.dir = self.out_root / "search" / "index"
        self.device = _device(self.cfg["device"])
        self._text = self._siglip = self._clip = None
        self.timings = {}

    # --- encoders
    @property
    def text(self):
        if self._text is None:
            self._text = ce.TextEmbedder("/nonexistent-mock" if self.cfg["mock"] else self.cfg["text_model"])
        return self._text

    @property
    def siglip(self):
        if self._siglip is None:
            self._siglip = _Mock(seed=0) if self.cfg["mock"] else _HFImageText(self.cfg["siglip_model"], self.device, True)
        return self._siglip

    @property
    def clipm(self):
        if self._clip is None:
            self._clip = _Mock(seed=1) if self.cfg["mock"] else _HFImageText(self.cfg["clip_model"], self.device, False)
        return self._clip

    def _hash(self, keys):
        return hashlib.md5(json.dumps([self.cfg[k] for k in keys], default=str).encode()).hexdigest()[:10]

    # --- build
    def build(self, force=False):
        """Load clips and selected frames, embed caption texts, and (cfg baselines) embed 1 frame/s."""
        self.dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        self.videos = self._videos()
        self.clips = self._load_clips()
        self.frames, self.frame_emb = self._load_selected_frames()
        self._map_frames_to_clips()
        texts = [ce.caption_text(r, ("summary", "actions", "people", "vehicles", "objects", "notable", "scene"))
                 for _, r in self.clips.iterrows()]
        self.text.fit(texts)   # TF-IDF fallback needs one vocabulary for captions and queries
        self.clip_text_emb = self.text.encode(texts) if texts else np.zeros((0, 1), np.float32)
        self.timings["index_s"] = time.perf_counter() - t0
        log(f"{len(self.clips)} clips, {len(self.frames)} selected frames, {len(self.videos)} videos; "
            f"text model: {self.text.name} ({'sentence model' if self.text.kind == 'model' else 'TF-IDF fallback'})")
        if self.cfg["baselines"]:
            self._build_baselines(force)
        return self

    def _videos(self):
        rows = []
        for sj in sorted(self.out_root.glob("*/*/summary_cpu.json")):
            s = json.loads(sj.read_text())
            if s.get("status_cpu") != "ok":
                continue
            rows.append({"dataset": s["dataset"], "video_id": s["video_id"], "camera_id": str(s.get("camera_id", "")),
                         "vdir": str(sj.parent), "path": s.get("path", ""), "duration_s": float(s.get("duration_s", 0)),
                         "start_wallclock": s.get("start_wallclock") or ""})
        return pd.DataFrame(rows)

    def _load_clips(self):
        c = ce.load_clips(self.out_root)
        if c.empty:
            return pd.DataFrame(columns=["dataset", "video_id", "camera_id", "vdir", "clip_id", "start_s", "end_s"])
        return c.reset_index(drop=True)

    def _load_selected_frames(self):
        rows, embs = [], []
        for v in self.videos.itertuples():
            ev = Path(v.vdir) / "events"
            if not (ev / "index_records.jsonl").exists() or not (ev / "index_embeddings.npy").exists():
                continue
            E = np.load(ev / "index_embeddings.npy").astype(np.float32)
            if E.ndim != 2 or len(E) == 0:
                continue
            for line in (ev / "index_records.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r["embedding_row"] >= len(E):
                    continue
                rows.append({"dataset": v.dataset, "video_id": v.video_id, "vdir": v.vdir, "camera_id": v.camera_id,
                             "frame_idx": r["frame_idx"], "timestamp_s": r["timestamp_s"]})
                embs.append(E[r["embedding_row"]])
        F = pd.DataFrame(rows)
        if not len(embs):
            return F, np.zeros((0, 1), np.float32)
        E = np.stack(embs)
        return F, E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)

    def _map_frames_to_clips(self):
        """frame -> row of the clip that contains it (-1: none, e.g. an idle heartbeat frame)."""
        self.frames["clip_row"] = -1
        self.clip_frames = [np.array([], int)] * len(self.clips)
        if not len(self.frames) or not len(self.clips):
            return
        for i, c in self.clips.iterrows():
            m = ((self.frames.vdir == c.vdir) & (self.frames.timestamp_s >= c.start_s)
                 & (self.frames.timestamp_s < c.end_s)).to_numpy()
            idx = np.flatnonzero(m)
            self.frames.loc[idx, "clip_row"] = i
            self.clip_frames[i] = idx

    def _build_baselines(self, force):
        h = self._hash(["siglip_model", "clip_model", "baseline_fps", "mock"])
        meta, fs, fc = self.dir / "baseline_frames.csv", self.dir / "baseline_siglip.npy", self.dir / "baseline_clip.npy"
        done = (self.dir / "baseline_hash.txt").read_text() if (self.dir / "baseline_hash.txt").exists() else ""
        if not force and done == h and meta.exists() and fs.exists() and fc.exists():
            self.base = pd.read_csv(meta, dtype={"video_id": str})
            self.base_siglip, self.base_clip = np.load(fs), np.load(fc)
            known = set(self.base.vdir)
            if set(self.videos.vdir) <= known:
                log(f"baselines: {len(self.base)} frames at {self.cfg['baseline_fps']}/s (cached)")
                return
        t0 = time.perf_counter()
        rows, S, C = [], [], []
        B = int(self.cfg["batch"])
        for i, v in enumerate(self.videos.itertuples(), 1):
            if not v.path or not Path(v.path).exists():
                log(f"baselines: video missing, skipped: {v.path}")
                continue
            info = video_info(v.path)
            step = max(1, int(round(info.fps / self.cfg["baseline_fps"])))
            idxs = list(range(0, info.frame_count, step))
            buf, buf_meta = [], []

            def flush():
                if buf:
                    S.append(self.siglip.images(buf)); C.append(self.clipm.images(buf))
                    rows.extend(buf_meta); buf.clear(); buf_meta.clear()
            for fi, frame in read_frames(v.path, idxs):
                buf.append(frame)
                buf_meta.append({"dataset": v.dataset, "video_id": v.video_id, "vdir": v.vdir, "camera_id": v.camera_id,
                                 "frame_idx": fi, "timestamp_s": round(fi / info.fps, 3)})
                if len(buf) >= B:
                    flush()
            flush()
            if i % 5 == 0 or i == len(self.videos):
                log(f"baselines: {i}/{len(self.videos)} videos embedded ({len(rows)} frames)")
        self.base = pd.DataFrame(rows)
        self.base_siglip = np.concatenate(S) if S else np.zeros((0, 1), np.float32)
        self.base_clip = np.concatenate(C) if C else np.zeros((0, 1), np.float32)
        self.base.to_csv(meta, index=False)
        np.save(fs, self.base_siglip.astype(np.float32)); np.save(fc, self.base_clip.astype(np.float32))
        (self.dir / "baseline_hash.txt").write_text(h)
        self.timings["baselines_s"] = time.perf_counter() - t0
        log(f"baselines: {len(self.base)} frames embedded in {self.timings['baselines_s'] / 60:.1f} min")

    # --- queries
    def encode(self, texts: list[str]) -> dict:
        """All query vectors at once: caption space, SigLIP text space, CLIP text space."""
        pre = self.cfg["query_prefix"] if self.text.kind == "model" else ""
        out = {"text": self.text.encode([pre + t for t in texts]), "siglip": self.siglip.texts(texts)}
        if len(self.frame_emb) and out["siglip"].shape[1] != self.frame_emb.shape[1]:
            raise ValueError(f"SigLIP text vectors are {out['siglip'].shape[1]}-d but the stage-4 frame embeddings are "
                             f"{self.frame_emb.shape[1]}-d: set siglip_model to the model stage 3 used "
                             f"(or re-run the gpu phase; mock runs need mock here too)")
        if self.cfg["baselines"]:
            out["clip"] = self.clipm.texts(texts)
        return out

    def rank(self, method, qv, i, dataset=None, video_id=None, camera_id=None, k=None):
        """Ranked results for query i (row of the encode() output) -> DataFrame with rank, video, clip and
        window spans, score. Filters: dataset, video, camera."""
        k = k or self.cfg["top_k"]
        if method in ("captions", "hybrid"):
            return self._rank_clips(method, qv, i, dataset, video_id, camera_id, k)
        if method == "siglip_selected":
            F, E, q = self.frames, self.frame_emb, qv["siglip"][i]
        elif method == "siglip_1fps":
            F, E, q = self.base, self.base_siglip, qv["siglip"][i]
        elif method == "clip_1fps":
            F, E, q = self.base, self.base_clip, qv["clip"][i]
        else:
            raise ValueError(method)
        return self._rank_frames(F, E, q, dataset, video_id, camera_id, k, method)

    def _filter(self, df, dataset, video_id, camera_id):
        m = np.ones(len(df), bool)
        if dataset is not None:
            m &= (df.dataset == dataset).to_numpy()
        if video_id is not None:
            m &= (df.video_id == video_id).to_numpy()
        if camera_id is not None:
            m &= (df.camera_id.astype(str) == str(camera_id)).to_numpy()
        return m

    def _duration(self, vdir):
        if not hasattr(self, "_dur"):
            self._dur = dict(zip(self.videos.vdir, self.videos.duration_s))
        return self._dur.get(vdir, 0.0)

    def _rank_clips(self, method, qv, i, dataset, video_id, camera_id, k):
        C = self.clips
        m = self._filter(C, dataset, video_id, camera_id)
        rows = np.flatnonzero(m)
        if not len(rows):
            return pd.DataFrame()
        cap = self.clip_text_emb[rows] @ qv["text"][i]
        # SigLIP: each clip's best selected frame
        fsc = self.frame_emb @ qv["siglip"][i] if len(self.frame_emb) else np.zeros(0)
        best_frame = np.full(len(rows), -1)
        sig = np.full(len(rows), -np.inf)
        for j, r in enumerate(rows):
            fr = self.clip_frames[r]
            if len(fr):
                b = fr[np.argmax(fsc[fr])]
                best_frame[j], sig[j] = b, fsc[b]
        if method == "captions":
            score = cap
        else:
            K = self.cfg["rrf_k"]
            r_cap = np.empty(len(rows)); r_cap[np.argsort(-cap)] = np.arange(1, len(rows) + 1)
            has = np.isfinite(sig)
            r_sig = np.full(len(rows), np.inf)
            r_sig[np.flatnonzero(has)[np.argsort(-sig[has])]] = np.arange(1, has.sum() + 1)
            score = 1 / (K + r_cap) + np.where(has, 1 / (K + r_sig), 0.0)
        order = np.argsort(-score)[:k]
        out = []
        for rank, j in enumerate(order, 1):
            c = C.iloc[rows[j]]
            t = self.frames.timestamp_s.iat[best_frame[j]] if best_frame[j] >= 0 else (c.start_s + c.end_s) / 2
            L = self.cfg["moment_s"]
            if c.end_s - c.start_s <= L:      # short clip: the window is the clip
                w0, w1 = c.start_s, c.end_s
            else:                             # L seconds around the best frame, kept inside the clip
                w0 = min(max(t - L / 2, c.start_s), c.end_s - L)
                w1 = w0 + L
            out.append({"rank": rank, "dataset": c.dataset, "video_id": c.video_id, "camera_id": c.camera_id,
                        "vdir": c.vdir, "clip_id": c.clip_id, "clip_start": c.start_s, "clip_end": c.end_s,
                        "win_start": round(w0, 2), "win_end": round(w1, 2), "best_t": round(float(t), 2),
                        "score": float(score[j]), "summary": c.get("summary", "")})
        return pd.DataFrame(out)

    def _rank_frames(self, F, E, q, dataset, video_id, camera_id, k, method):
        if not len(F):
            return pd.DataFrame()
        m = self._filter(F, dataset, video_id, camera_id)
        rows = np.flatnonzero(m)
        if not len(rows):
            return pd.DataFrame()
        s = E[rows] @ q
        order = rows[np.argsort(-s)]
        sc = dict(zip(rows, s))
        kept, out = [], []
        nms = self.cfg["nms_s"]
        for r in order:   # non-maximum suppression: one result per nms_s seconds of a video
            f = F.iloc[r]
            if any(f.vdir == kv and abs(f.timestamp_s - kt) < nms for kv, kt in kept):
                continue
            kept.append((f.vdir, f.timestamp_s))
            t = float(f.timestamp_s)
            w0, w1 = window(t, self.cfg["moment_s"], self._duration(f.vdir))
            cr = int(f["clip_row"]) if "clip_row" in F.columns else -1
            if cr >= 0:
                c0, c1, cid = self.clips.start_s.iat[cr], self.clips.end_s.iat[cr], self.clips.clip_id.iat[cr]
            else:   # baselines / frames outside clips: the window doubles as the "clip"
                c0, c1, cid = w0, w1, ""
            out.append({"rank": len(out) + 1, "dataset": f.dataset, "video_id": f.video_id, "camera_id": f.camera_id,
                        "vdir": f.vdir, "clip_id": cid, "clip_start": c0, "clip_end": c1,
                        "win_start": round(w0, 2), "win_end": round(w1, 2), "best_t": round(t, 2),
                        "score": float(sc[r]), "summary": ""})
            if len(out) >= k:
                break
        return pd.DataFrame(out)

    def search(self, query: str, k=10, method="hybrid", dataset=None, video_id=None, camera_id=None):
        """One query, ranked results. For the notebook's 'try a query' cell and the latency probe."""
        qv = self.encode([query])
        return self.rank(method, qv, 0, dataset, video_id, camera_id, k)


# ---------------------------------------------------------------------------------------------------
# Queries with ground truth
# ---------------------------------------------------------------------------------------------------
def make_queries(index: SearchIndex) -> pd.DataFrame:
    """UCA: one query per sentence (answer = its own span in its video). MEVA: one query per activity type
    (answers = every instance, in any MEVA video). Other datasets: one query per label, like MEVA."""
    rows = []
    for ds, vids in index.videos.groupby("dataset"):
        per_label = {}
        for v in vids.itertuples():
            gt_path = Path(v.vdir) / "gt.csv"
            if not gt_path.exists():
                continue
            gt = pd.read_csv(gt_path)
            for g in gt.itertuples():
                span = [v.video_id, round(float(g.start_s), 2), round(float(g.end_s), 2)]
                if ds == "uca":
                    rows.append({"dataset": ds, "source": "uca_sentence", "text": str(g.label), "label": "",
                                 "video_id": v.video_id, "gt": [span]})
                else:
                    per_label.setdefault(str(g.label), []).append(span)
        for label, spans in sorted(per_label.items()):
            text = MEVA_QUERIES.get(label, label.replace("_", " "))
            rows.append({"dataset": ds, "source": f"{ds}_activity", "text": text, "label": label,
                         "video_id": "", "gt": spans})
    q = pd.DataFrame(rows)
    if q.empty:
        return pd.DataFrame(columns=["query_id", "dataset", "source", "text", "label", "video_id", "n_gt", "gt"])
    q.insert(0, "query_id", [f"{d}_{i:03d}" for d, i in zip(q.dataset, q.groupby("dataset").cumcount() + 1)])
    q["n_gt"] = q["gt"].apply(len)
    q["gt"] = q["gt"].apply(json.dumps)
    return q


def load_or_make_queries(index: SearchIndex, path=None, remake=False) -> pd.DataFrame:
    path = Path(path or index.out_root / "search" / "queries.csv")
    if path.exists() and not remake:
        q = pd.read_csv(path, keep_default_na=False, dtype={"video_id": str})
    else:
        q = make_queries(index)
        path.parent.mkdir(parents=True, exist_ok=True)
        q.to_csv(path, index=False)
    q["gt_spans"] = q["gt"].apply(json.loads)
    return q


# ---------------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------------
def score_results(res: pd.DataFrame, spans, ious, ks) -> dict:
    """First rank with a hit, per span kind and IoU threshold, plus lenient (any overlap)."""
    out = {}
    for kind, a, b in (("clip", "clip_start", "clip_end"), ("window", "win_start", "win_end")):
        best_iou, any_ov = [], []
        for r in res.itertuples():
            same = [(s0, s1) for vid, s0, s1 in spans if vid == r.video_id]
            best_iou.append(max([iou(getattr(r, a), getattr(r, b), s0, s1) for s0, s1 in same], default=0.0))
            any_ov.append(any(overlap(getattr(r, a), getattr(r, b), s0, s1) > 0 for s0, s1 in same))
        for t in ious:
            hits = [i + 1 for i, v in enumerate(best_iou) if v >= t]
            out[f"{kind}_first_hit@{t}"] = hits[0] if hits else np.nan
        hits = [i + 1 for i, v in enumerate(any_ov) if v]
        out[f"{kind}_first_hit_lenient"] = hits[0] if hits else np.nan
        out[f"{kind}_best_iou_top1"] = best_iou[0] if best_iou else 0.0
    return out


def reachable(index: SearchIndex, spans) -> bool:
    """Does any answer overlap any clip? (No: the gating lost it, no caption can find it.)"""
    C = index.clips
    for vid, s0, s1 in spans:
        c = C[C.video_id == vid]
        if ((np.minimum(c.end_s, s1) - np.maximum(c.start_s, s0)) > 0).any():
            return True
    return False


def evaluate(index: SearchIndex, queries: pd.DataFrame, methods=None, log_every=50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every query x method x setting. Returns (results: top-k rows, per_query: first-hit ranks)."""
    cfg = index.cfg
    methods = [m for m in (methods or METHODS) if cfg["baselines"] or m not in ("siglip_1fps", "clip_1fps")]
    t0 = time.perf_counter()
    qv = index.encode(list(queries.text))
    index.timings["encode_queries_s"] = time.perf_counter() - t0
    res_rows, pq_rows = [], []
    for i, q in enumerate(queries.itertuples()):
        reach = reachable(index, q.gt_spans)
        settings = [("corpus", None)] + ([("video", q.video_id)] if q.video_id else [])
        for setting, vid in settings:
            for m in methods:
                tm = time.perf_counter()
                res = index.rank(m, qv, i, dataset=q.dataset, video_id=vid)
                ms = 1000 * (time.perf_counter() - tm)
                sc = score_results(res, q.gt_spans, cfg["iou"], cfg["ks"]) if len(res) else {}
                pq_rows.append({"query_id": q.query_id, "dataset": q.dataset, "source": q.source, "label": q.label,
                                "text": q.text, "setting": setting, "method": m, "reachable": reach,
                                "n_results": len(res), "rank_ms": ms, **sc})
                if len(res):
                    res = res.assign(query_id=q.query_id, setting=setting, method=m)
                    res_rows.append(res.drop(columns=["summary"]))
        if log_every and (i + 1) % log_every == 0:
            log(f"{i + 1}/{len(queries)} queries")
    results = pd.concat(res_rows, ignore_index=True) if res_rows else pd.DataFrame()
    return results, pd.DataFrame(pq_rows)


def metrics(per_query: pd.DataFrame, ks=(1, 5, 10), ious=(0.3, 0.5)) -> pd.DataFrame:
    """Recall@K per dataset x setting x method (window spans for IoU, clip spans for lenient), MRR."""
    rows = []
    for (ds, setting, m), g in per_query.groupby(["dataset", "setting", "method"], sort=False):
        r = {"dataset": ds, "setting": setting, "method": m, "queries": len(g),
             "reachable %": 100 * g.reachable.mean()}
        for t in ious:
            for k in ks:
                r[f"R@{k} IoU{t}"] = 100 * (g[f"window_first_hit@{t}"] <= k).mean()
        for k in ks:
            r[f"R@{k} lenient"] = 100 * (g["clip_first_hit_lenient"] <= k).mean()
        for t in ious:
            r[f"R@5 IoU{t} (clip span)"] = 100 * (g[f"clip_first_hit@{t}"] <= 5).mean()
        gr = g[g.reachable]   # only the queries the gating kept: is search itself any good?
        r["R@5 lenient | reachable"] = 100 * (gr["clip_first_hit_lenient"] <= 5).mean() if len(gr) else np.nan
        t0 = ious[0]
        r[f"MRR IoU{t0}"] = float((1 / g[f"window_first_hit@{t0}"]).fillna(0).mean())
        r["rank ms (median)"] = float(g.rank_ms.median())
        rows.append(r)
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS)}
    return out.sort_values(["dataset", "setting", "method"], key=lambda s: s.map(order) if s.name == "method" else s) \
        .reset_index(drop=True).round(2)


def latency_probe(index: SearchIndex, queries: pd.DataFrame, n=20, methods=None, seed=0) -> pd.DataFrame:
    """End-to-end time of single queries (encode + rank), like a user typing one."""
    rng = np.random.default_rng(seed)
    pick = queries.iloc[rng.choice(len(queries), size=min(n, len(queries)), replace=False)]
    rows = []
    for m in methods or METHODS:
        if not index.cfg["baselines"] and m in ("siglip_1fps", "clip_1fps"):
            continue
        for q in pick.itertuples():
            t0 = time.perf_counter()
            index.search(q.text, method=m, dataset=q.dataset)
            rows.append({"method": m, "ms": 1000 * (time.perf_counter() - t0)})
    d = pd.DataFrame(rows)
    return d.groupby("method").ms.agg(median="median", p95=lambda s: s.quantile(0.95)).round(1) \
        .reindex([m for m in METHODS if m in set(d.method)])


def save(index: SearchIndex, queries, results, per_query, met, lat=None):
    """outputs/batch/search/: queries.csv, results.csv, per_query.csv, metrics.csv, latency.csv, search_run.json."""
    d = index.out_root / "search"
    d.mkdir(parents=True, exist_ok=True)
    queries.drop(columns=["gt_spans"], errors="ignore").to_csv(d / "queries.csv", index=False)
    results.to_csv(d / "results.csv", index=False)
    per_query.to_csv(d / "per_query.csv", index=False)
    met.to_csv(d / "metrics.csv", index=False)
    if lat is not None:
        lat.to_csv(d / "latency.csv")
    run = {"config": index.cfg, "device": index.device, "timings": index.timings,
           "text_embedder": f"{index.text.name} ({'sentence model' if index.text.kind == 'model' else 'TF-IDF fallback'})",
           "clips": len(index.clips), "selected_frames": len(index.frames),
           "baseline_frames": len(getattr(index, "base", [])), "queries": len(queries),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / "search_run.json").write_text(json.dumps(run, indent=1, default=str))
    return d


# ---------------------------------------------------------------------------------------------------
# Pictures for explore.ipynb section 8
# ---------------------------------------------------------------------------------------------------
_paths = {}


def frame_at(vdir, t):
    """The video frame at t seconds of the video whose output folder is vdir (BGR), or None."""
    if vdir not in _paths:
        sj = Path(vdir) / "summary_cpu.json"
        _paths[vdir] = json.loads(sj.read_text()).get("path", "") if sj.exists() else ""
    p = _paths[vdir]
    if not p or not Path(p).exists():
        return None
    info = video_info(p)
    fi = min(max(0, int(round(t * info.fps))), max(0, info.frame_count - 1))
    for _, f in read_frames(p, [fi]):
        return f
    return None


def result_strip(res: pd.DataFrame, spans=None, n=5, width=240):
    """Top-n results side by side (frame at each result's best moment). Green border = the result overlaps
    an answer in `spans` ([video_id, start, end] list), red = it doesn't, grey = no ground truth."""
    import cv2
    tiles = []
    for r in res.head(n).itertuples():
        f = frame_at(r.vdir, r.best_t)
        if f is None:
            f = np.zeros((135, 240, 3), np.uint8)
        f = cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])))
        if spans is None:
            col = (160, 160, 160)
        else:
            hit = any(vid == r.video_id and overlap(r.clip_start, r.clip_end, s0, s1) > 0 for vid, s0, s1 in spans)
            col = (60, 180, 60) if hit else (40, 40, 220)
        f = cv2.copyMakeBorder(f, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=col)
        label = f"#{r.rank} {r.video_id[:18]} {r.best_t:.0f}s"
        cv2.rectangle(f, (4, 4), (f.shape[1] - 4, 22), (0, 0, 0), -1)
        cv2.putText(f, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(f)
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    return np.hstack(tiles)


def load_outputs(out_root):
    """Everything search.ipynb saved, as a dict of DataFrames (missing files -> None)."""
    d = Path(out_root) / "search"
    out = {}
    for name in ("queries", "results", "per_query", "metrics", "latency", "demo_results"):
        p = d / f"{name}.csv"
        out[name] = pd.read_csv(p, keep_default_na=False if name == "queries" else True,
                                dtype={"video_id": str, "clip_id": str}) if p.exists() else None
    if out["queries"] is not None:
        out["queries"]["gt_spans"] = out["queries"]["gt"].apply(json.loads)
    out["run"] = json.loads((d / "search_run.json").read_text()) if (d / "search_run.json").exists() else {}
    return out
