"""Find the videos and annotations of our two datasets, pick a ~1 h subset of each, load ground truth.

Datasets (on the team VM, under DATA_ROOT):

  meva/videos/2018-03-13.15-50-00.15-55-00.school.G328.r13.avi      5-min clips from fixed cameras
  meva/annotations/.../2018-03-13.15-50-00.15-55-00.school.G328.activities.yml   (only some clips have one)
  uca/dev_videos/Burglary090_x264.mp4                               UCF-Crime videos (our dev set)
  uca/annotations/UCF Annotation/json/UCFCrime_{Train,Val,Test}.json  UCA sentences with start/end times

What the ground truth means is different for the two datasets:

- MEVA `*.activities.yml`: one line per activity (person_carries_heavy_object, vehicle_starts, ...), with a
  frame range `tsr0: [start_frame, end_frame]`. Frame numbers, not seconds, so we divide by the video's fps.
  Only ~37 activity types are labelled; a person just walking past usually is not. So the labels say
  "these moments must not be missed", not "everything else is empty".
- UCA: every video is described by sentences with [start, end] seconds that follow each other back to back
  and cover the whole video, quiet parts included. So UCA can't say which parts are "inactive". What it can
  check: does every sentence's time span get at least one keyframe? A span without a keyframe can never be
  found by a search over keyframes. (Later, the same sentences are the text queries for Recall@K.)

Command line:  python datasets.py            -> prints what is on disk (videos, hours, annotations)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("/mnt/5T_storage/clipsExtraction/datasets")

# 2018-03-13.15-50-00.15-55-00.school.G328.r13   (date . start . end . site . camera [. release])
MEVA_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.([^.]+)\.([^.]+?)(?:\.(r\d+))?$")
UCA_NAME = re.compile(r"^(.*?)_?(\d+)_x264$")       # Burglary090_x264 -> Burglary; Normal_Videos_015_x264 -> Normal_Videos
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov"}
SELECTION_COLS = ["dataset", "video_id", "path", "camera_id", "site", "start_wallclock", "duration_s",
                  "category", "split", "gt_source", "gt_match", "n_gt"]


# ---------------------------------------------------------------------------------------------------
# MEVA
# ---------------------------------------------------------------------------------------------------
def parse_meva_name(stem: str) -> dict | None:
    """'2018-03-13.15-50-00.15-55-00.school.G328.r13' -> date, start/end datetimes, site, camera, key.
    key drops the release suffix (.r13), because the annotation files are named without it."""
    m = MEVA_NAME.match(stem)
    if not m:
        return None
    date, t0, t1, site, cam, _rel = m.groups()
    start = datetime.strptime(f"{date} {t0}", "%Y-%m-%d %H-%M-%S")
    end = datetime.strptime(f"{date} {t1}", "%Y-%m-%d %H-%M-%S")
    if end < start:                       # clip runs past midnight
        end += timedelta(days=1)
    return {"date": date, "start": start, "end": end, "site": site, "camera": cam,
            "key": f"{date}.{t0}.{t1}.{site}.{cam}"}


def meva_annotation_index(meva_dir: Path) -> dict[str, Path]:
    """{key: path of its .activities.yml}. Searches meva/annotations and meva/meva-data-repo/annotation."""
    index = {}
    for sub in ("annotations", "meva-data-repo/annotation"):
        d = meva_dir / sub
        if not d.exists():
            continue
        for p in d.rglob("*.activities.yml"):
            info = parse_meva_name(p.name[: -len(".activities.yml")])
            if info:
                index.setdefault(info["key"], p)
    return index


def _match_annotation(info: dict, index: dict[str, Path], by_cam: dict, tol_s: float = 5.0):
    """Exact name match first; else same date + camera with a start time within tol_s seconds."""
    if info["key"] in index:
        return index[info["key"]], "exact"
    for start, p in by_cam.get((info["date"], info["camera"]), []):
        if abs((start - info["start"]).total_seconds()) <= tol_s:
            return p, "fuzzy"
    return None, "none"


def meva_inventory(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """One row per MEVA video on disk, with its duration (from the file name) and its annotation file."""
    meva = Path(data_root) / "meva"
    index = meva_annotation_index(meva)
    by_cam: dict = {}
    for key, p in index.items():
        i = parse_meva_name(key)
        by_cam.setdefault((i["date"], i["camera"]), []).append((i["start"], p))
    rows = []
    for p in sorted((meva / "videos").glob("*")):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        info = parse_meva_name(p.stem)
        if info is None:
            continue
        ann, how = _match_annotation(info, index, by_cam)
        rows.append({"dataset": "meva", "video_id": p.stem, "path": str(p), "camera_id": info["camera"],
                     "site": info["site"], "start_wallclock": info["start"].isoformat(),
                     "duration_s": (info["end"] - info["start"]).total_seconds(),
                     "category": info["site"], "split": "", "gt_source": str(ann) if ann else "",
                     "gt_match": how, "n_gt": count_meva_activities(ann) if ann else 0})
    return pd.DataFrame(rows, columns=SELECTION_COLS)


def _read_kpf_lines(path: Path) -> list:
    """A KPF .yml file is a YAML list of one-line dicts: `- {'act': {...}}`. Use PyYAML if installed,
    otherwise read each `- {...}` line as a Python literal (that's exactly what those lines are)."""
    try:
        import yaml
        items = yaml.safe_load(Path(path).read_text())
        return items if isinstance(items, list) else []
    except ImportError:
        pass
    items = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("- {"):
            try:
                items.append(ast.literal_eval(line[2:]))
            except (ValueError, SyntaxError):
                continue
    return items


def count_meva_activities(path) -> int:
    return sum(1 for it in _read_kpf_lines(Path(path)) if isinstance(it, dict) and ("act" in it or "activity" in it))


def load_meva_gt(path, fps: float) -> pd.DataFrame:
    """Activities -> DataFrame start_s, end_s, label (+ start_frame, end_frame, actor ids)."""
    rows = []
    for it in _read_kpf_lines(Path(path)):
        act = it.get("act") or it.get("activity") if isinstance(it, dict) else None
        if not isinstance(act, dict):
            continue
        names = act.get("act2") or act.get("act3") or {}
        label = max(names, key=names.get) if isinstance(names, dict) and names else str(names)
        spans = act.get("timespan") or []
        fr = spans[0].get("tsr0") if spans and isinstance(spans[0], dict) else None
        if not fr or len(fr) < 2:
            continue
        actors = [a.get("id1") for a in act.get("actors", []) if isinstance(a, dict)]
        rows.append({"start_s": fr[0] / fps, "end_s": fr[1] / fps, "label": label,
                     "start_frame": int(fr[0]), "end_frame": int(fr[1]), "actors": " ".join(map(str, actors))})
    gt = pd.DataFrame(rows, columns=["start_s", "end_s", "label", "start_frame", "end_frame", "actors"])
    return gt.sort_values("start_s").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------
# UCA (UCF-Crime Annotation)
# ---------------------------------------------------------------------------------------------------
_UCA_CACHE: dict[str, dict] = {}


def uca_annotations(data_root: Path = DATA_ROOT) -> dict[str, dict]:
    """{video_id: {"split", "duration", "timestamps", "sentences"}} from UCFCrime_{Train,Val,Test}.json.
    (revised_en_UCFCrime_Train.json is skipped: same videos as Train, reworded sentences.)"""
    key = str(data_root)
    if key in _UCA_CACHE:
        return _UCA_CACHE[key]
    out = {}
    ann_dir = Path(data_root) / "uca" / "annotations"
    for split in ("Test", "Val", "Train"):     # if a video were listed twice, Test wins
        for p in ann_dir.rglob(f"UCFCrime_{split}.json"):
            for vid, a in json.loads(p.read_text()).items():
                out.setdefault(vid, {**a, "split": split.lower(), "file": str(p)})
    _UCA_CACHE[key] = out
    return out


def uca_category(video_id: str) -> str:
    m = UCA_NAME.match(video_id)
    return m.group(1).rstrip("_") if m else video_id


def uca_inventory(data_root: Path = DATA_ROOT, video_dir: str = "dev_videos") -> pd.DataFrame:
    """One row per UCA video in uca/<video_dir>/, with split and duration from the annotations
    (probed from the file when a video has no annotation)."""
    from video_utils import video_info
    ann = uca_annotations(data_root)
    rows = []
    for p in sorted((Path(data_root) / "uca" / video_dir).glob("*")):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        a = ann.get(p.stem)
        dur = float(a["duration"]) if a and "duration" in a else video_info(p).duration_s
        rows.append({"dataset": "uca", "video_id": p.stem, "path": str(p), "camera_id": p.stem, "site": "",
                     "start_wallclock": "", "duration_s": dur, "category": uca_category(p.stem),
                     "split": a["split"] if a else "", "gt_source": a["file"] if a else "",
                     "gt_match": "exact" if a else "none", "n_gt": len(a.get("sentences", [])) if a else 0})
    return pd.DataFrame(rows, columns=SELECTION_COLS)


_JSON_CACHE: dict[str, dict] = {}


def load_uca_gt(video_id: str, json_path, duration_s: float | None = None) -> pd.DataFrame:
    """UCA sentences of one video (from the JSON file its selection row points to) -> DataFrame start_s,
    end_s, label (= the sentence). Clipped to the real video length."""
    key = str(json_path)
    if key not in _JSON_CACHE:
        _JSON_CACHE[key] = json.loads(Path(json_path).read_text())
    a = _JSON_CACHE[key].get(video_id)
    if not a:
        return pd.DataFrame(columns=["start_s", "end_s", "label"])
    gt = pd.DataFrame([{"start_s": float(s), "end_s": float(e), "label": txt}
                       for (s, e), txt in zip(a["timestamps"], a["sentences"])])
    if duration_s:
        gt["end_s"] = gt["end_s"].clip(upper=duration_s)
        gt = gt[gt.end_s > gt.start_s]
    return gt.reset_index(drop=True)


def load_gt(row, fps: float, duration_s: float | None = None) -> pd.DataFrame | None:
    """Ground truth for one selection row (None if the video has none). Uses the row's gt_source file."""
    src = row.get("gt_source") or ""
    if not src:
        return None
    if row["dataset"] == "meva":
        return load_meva_gt(src, fps)
    if row["dataset"] == "uca":
        gt = load_uca_gt(row["video_id"], src, duration_s)
        return gt if len(gt) else None
    return None


# ---------------------------------------------------------------------------------------------------
# Picking ~1 h per dataset
# ---------------------------------------------------------------------------------------------------
def select_meva(inv: pd.DataFrame, target_min: float = 60, prefer_gt: bool = True,
                cameras: list[str] | None = None, slot_min: int = 5) -> pd.DataFrame:
    """Whole time slots (every camera recording in the same 5-min window), so the subset is multi-camera.
    Slots with annotated clips come first (if prefer_gt), then in time order, until target_min is reached.
    With 4 cameras that's 3 slots = 15 min of the same scene from 4 angles."""
    df = inv.copy()
    if cameras:
        df = df[df.camera_id.isin(cameras)]
    if df.empty:
        return df
    t = pd.to_datetime(df.start_wallclock)
    df["slot"] = t.dt.floor(f"{slot_min}min")
    slots = df.groupby("slot").agg(n_gt=("gt_match", lambda s: int((s != "none").sum())),
                                   minutes=("duration_s", lambda s: s.sum() / 60)).reset_index()
    slots = slots.sort_values(["n_gt", "slot"], ascending=[False, True]) if prefer_gt else slots.sort_values("slot")
    keep, total = [], 0.0
    for s in slots.itertuples():
        if total >= target_min:
            break
        keep.append(s.slot)
        total += s.minutes
    return df[df.slot.isin(keep)].sort_values(["slot", "camera_id", "start_wallclock"]).drop(columns="slot")


def select_uca(inv: pd.DataFrame, target_min: float = 60, prefer_split: str | None = "test",
               max_video_min: float = 10, seed: int = 0) -> pd.DataFrame:
    """A spread over categories: round-robin (one video per category per round, random order within a
    category, `prefer_split` first), skipping videos without annotations or longer than max_video_min,
    until target_min is reached."""
    df = inv[(inv.gt_match != "none") & (inv.duration_s <= max_video_min * 60)].copy()
    if df.empty:
        return df
    rng = np.random.default_rng(seed)
    df["_r"] = rng.random(len(df))
    df["_pref"] = (df.split != prefer_split) if prefer_split else False
    queues = {c: list(g.sort_values(["_pref", "_r"]).index) for c, g in df.groupby("category")}
    keep, total = [], 0.0
    while total < target_min and any(queues.values()):
        for c in sorted(queues):
            if queues[c] and total < target_min:
                i = queues[c].pop(0)
                keep.append(i)
                total += df.at[i, "duration_s"] / 60
    return df.loc[keep].drop(columns=["_r", "_pref"]).sort_values(["category", "video_id"])


def make_selection(data_root: Path = DATA_ROOT, meva_min: float = 60, uca_min: float = 60,
                   uca_video_dir: str = "dev_videos", meva_cameras: list[str] | None = None) -> pd.DataFrame:
    parts = []
    if meva_min > 0:
        parts.append(select_meva(meva_inventory(data_root), meva_min, cameras=meva_cameras))
    if uca_min > 0:
        parts.append(select_uca(uca_inventory(data_root, uca_video_dir), uca_min))
    parts = [p for p in parts if len(p)]
    sel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SELECTION_COLS)
    return sel[SELECTION_COLS]


def load_or_make_selection(path, data_root: Path = DATA_ROOT, remake: bool = False, **kw) -> pd.DataFrame:
    """Reuse selection.csv if it exists (edit it by hand to change the videos), else make and save it."""
    path = Path(path)
    if path.exists() and not remake:
        return pd.read_csv(path, keep_default_na=False)
    sel = make_selection(data_root, **kw)
    sel.to_csv(path, index=False)
    return pd.read_csv(path, keep_default_na=False)


def describe(inv: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: videos, hours, how many have ground truth."""
    if inv.empty:
        return pd.DataFrame()
    return inv.groupby("dataset").agg(videos=("video_id", "size"), hours=("duration_s", lambda s: s.sum() / 3600),
                                      with_gt=("gt_match", lambda s: int((s != "none").sum())),
                                      gt_items=("n_gt", "sum")).round(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="What is on disk: videos, hours, annotations.")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    a = ap.parse_args()
    root = Path(a.data_root)
    meva, uca = meva_inventory(root), uca_inventory(root)
    inv = pd.concat([meva, uca], ignore_index=True)
    print(describe(inv).to_string(), "\n")
    if len(meva):
        print("MEVA cameras:", meva.camera_id.value_counts().to_dict())
        print("MEVA clips shorter than 4.5 min:", int((meva.duration_s < 270).sum()))
        print("MEVA annotation matches:", meva.gt_match.value_counts().to_dict())
    if len(uca):
        print("UCA splits:", uca.split.replace("", "(no annotation)").value_counts().to_dict())
        print("UCA categories:", uca.category.value_counts().to_dict())
        print(f"UCA durations (min): median {uca.duration_s.median() / 60:.1f}, max {uca.duration_s.max() / 60:.1f}")


if __name__ == "__main__":
    main()
