"""Measure gates and keyframes against hand-labelled ground truth (logic from the v2 pipeline).

Ground truth = CSV with one row per event: start_s,end_s[,label] (seconds from the start of the video).

  gate_table      per gate: frame precision / recall / F1 (share of sampled moments labelled correctly
                  as active), event recall (share of GT events a segment covers), % of video active
  keyframe_hits   per keyframe set: share of GT events that contain at least one keyframe. This is the
                  number that matters downstream: an event without a keyframe is never captioned or indexed.
  gate_agreement  IoU of active time between every pair of gates (no ground truth needed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from motion_gating import active_fraction, active_mask


def load_gt(path) -> pd.DataFrame:
    gt = pd.read_csv(path)
    missing = {"start_s", "end_s"} - set(gt.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing} (need start_s,end_s[,label])")
    return gt


def _covered(a, b, segs, iou_thr):
    """A GT event [a, b] counts as found if one segment covers > half of it, or overlaps it with IoU >= iou_thr."""
    for s, e in zip(segs.start_s, segs.end_s):
        inter = max(0.0, min(b, e) - max(a, s))
        if inter / max(1e-9, b - a) > 0.5 or inter / max(1e-9, max(b, e) - min(a, s)) >= iou_thr:
            return True
    return False


def gate_table(gt: pd.DataFrame, gates: dict, times: np.ndarray, duration_s: float, iou_thr=0.3) -> pd.DataFrame:
    """gates: {name: {"segments": DataFrame}} (motion_gating results) or {name: segments DataFrame}."""
    gt_mask = active_mask(times, gt)
    rows = {}
    for name, g in gates.items():
        segs = g["segments"] if isinstance(g, dict) else g
        pm = active_mask(times, segs)
        tp = (pm & gt_mask).sum()
        prec, rec = tp / max(1, pm.sum()), tp / max(1, gt_mask.sum())
        rows[name] = {"frame_precision": prec, "frame_recall": rec,
                      "frame_f1": 2 * prec * rec / max(1e-9, prec + rec),
                      "event_recall": np.mean([_covered(a, b, segs, iou_thr) for a, b in zip(gt.start_s, gt.end_s)])
                      if len(gt) else np.nan,
                      "active_pct": 100 * active_fraction(segs, duration_s)}
    return pd.DataFrame(rows).T


def keyframe_hits(gt: pd.DataFrame, keyframe_times: dict) -> pd.DataFrame:
    """keyframe_times: {name: array of keyframe timestamps (s)}."""
    rows = {}
    for name, kt in keyframe_times.items():
        kt = np.asarray(kt, float)
        hit = [bool(np.any((kt >= a) & (kt <= b))) for a, b in zip(gt.start_s, gt.end_s)]
        rows[name] = {"keyframes": len(kt), "gt_events_with_keyframe": float(np.mean(hit)) if hit else np.nan}
    return pd.DataFrame(rows).T


def gate_agreement(gates: dict, times: np.ndarray) -> pd.DataFrame:
    masks = {m: active_mask(times, g["segments"] if isinstance(g, dict) else g) for m, g in gates.items()}
    return pd.DataFrame({a: {b: (masks[a] & masks[b]).sum() / max(1, (masks[a] | masks[b]).sum()) for b in masks}
                         for a in masks}).T
