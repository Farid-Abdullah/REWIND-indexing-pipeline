"""Optional third motion gate: DINOv2 patch features (from the v2 pipeline, rewind_pipeline.py).

Adds a `score_deep` column to the scores from motion_gating.scan_video, so the deep gate can be
thresholded, segmented and fused exactly like diff and MOG2 (motion_gating.segment_methods / add_fused).

How it works (rewind_pipeline.DeepFeatureGate): every frame is cut into 14x14-pixel patches and DINOv2
turns each patch into a feature vector. Each patch keeps a slowly updated "background" vector; a patch
counts as moving when it differs from its background much more than usual. Score = share of patches
moving. Runs at `deep_fps` (2/s) and is held constant in between.

Needs torch + transformers and, realistically, a GPU. If the model can't load, scan_deep returns None
and the notebook simply leaves the deep gate out.
Known weakness (seen on cafe_demo.mp4): the first frames become the background, so people present at
the start count as "change" until the slow update absorbs them (about a minute).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import rewind_pipeline as rp
from video_utils import read_frames


def scan_deep(video_path, scores: pd.DataFrame, cfg: dict, models: "rp.Models",
              cache_dir: str | Path | None = None) -> tuple[np.ndarray | None, dict]:
    """Deep-gate score for every row of `scores` (same samples as stage 1).

    cfg:    the v2 config (rp.DEFAULT_CONFIG plus overrides); uses cfg["scan"]["deep_*"], batch_size,
            ignore_polygons and cfg["runtime"]["mock_models"]
    models: rp.Models(cfg); models.patch() loads DINOv2 (or a mock)
    Returns (score array or None if the model is unavailable, stats dict). Cached in cache_dir.
    """
    s = cfg["scan"]
    video_path = Path(video_path)
    keys = ["deep_fps", "deep_model", "deep_input_h", "deep_z", "deep_min_dist", "deep_alpha_bg",
            "deep_alpha_fg", "deep_alpha_stats", "deep_warmup", "ignore_polygons"]
    ident = {"video": video_path.name, "size": video_path.stat().st_size, "n": len(scores),
             "mock": cfg["runtime"]["mock_models"], **{k: s[k] for k in keys}}
    key = hashlib.sha1(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"deep_{key}.npz" if cache_dir else None
    if cache and cache.exists():
        z = np.load(cache)
        return z["score"], {"from_cache": True, "deep_s": 0.0}

    enc = models.patch()
    if enc is None:
        return None, {"skipped": models.failed.get("deep_patch", "model unavailable")}

    period = float(np.median(np.diff(scores["timestamp_s"])))
    every = max(1, round((1.0 / period) / s["deep_fps"]))     # 5 samples/s / 2 per s -> every 2nd sample
    rows = np.arange(0, len(scores), every)
    fidx = scores["frame_idx"].to_numpy()[rows]
    gate, out, buf, buf_rows = None, np.full(len(scores), np.nan), [], []
    t0 = time.perf_counter()

    def flush():
        for r, f in zip(buf_rows, enc(buf)):
            out[r] = gate.update(f)[0]
        buf.clear()
        buf_rows.clear()

    for (fi, frame), r in zip(read_frames(video_path, fidx), rows):
        if gate is None:
            small = rp.resize_w(frame, 320)
            gate = rp.DeepFeatureGate(s, rp.polygon_mask(small.shape[:2], s["ignore_polygons"]))
        buf.append(frame)
        buf_rows.append(r)
        if len(buf) >= s["batch_size"]:
            flush()
    if buf:
        flush()
    elapsed = time.perf_counter() - t0
    score = pd.Series(out).ffill().fillna(0.0).to_numpy()   # hold each value until the next deep sample
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, score=score)
    return score, {"from_cache": False, "deep_s": elapsed, "deep_frames": int(len(rows)),
                   "deep_frames_per_s": len(rows) / max(elapsed, 1e-9)}
