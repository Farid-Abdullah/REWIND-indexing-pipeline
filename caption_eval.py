"""How good are the captions? Three checks on what the pipeline already saved (no re-run needed).

Used by explore.ipynb section 6. Each function reads outputs/batch/ and returns tables; nothing here calls the VLM.

1. detector_crosscheck   Automatic, all clips. Compares what a clip caption mentions (people, vehicles, bags, boxes,
                         ...) with what YOLO-World tracked in the same seconds (events/tracks.csv). Mentioned but
                         never tracked = likely hallucination; tracked for >= 1 s but not mentioned = likely miss.
                         Only covers the detector's 13 classes (no actions, no colours), and the detector errs too,
                         so it is a cheap screen, not ground truth.
2. label_similarity      Automatic. Pairs each ground-truth label (UCA sentence; MEVA activity name) with the clip
                         caption covering the same seconds and measures text similarity, against captions of other
                         videos as a baseline. Also: how often the most similar caption of all is the right clip
                         (top-1 / top-5) = caption-only search on the labels.
3. make_review_sheets    The hand check the supervisor asked for: random clips split over N reviewers, an HTML page
   + review_results      per reviewer (frames next to the caption) and a CSV to fill in. review_results turns the
                         filled-in CSVs into the hallucination and miss rates, and checks how well (1) agrees.

TextEmbedder (sentence embeddings for 2) is shared with the search.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

if Path("/mnt/5T_storage").exists():  # team VM: model downloads go to the 5T disk
    os.environ.setdefault("HF_HOME", "/mnt/5T_storage/hf_cache")

TEXT_MODEL = "BAAI/bge-small-en-v1.5"   # ~130 MB sentence-embedding model; any BERT-style model folder works

# detector classes (gpu_stages detector_vocab) grouped into what a caption would say, and the words for each group
CATEGORIES = {
    "person": (["person"], r"\b(person|people|man|men|woman|women|individuals?|child|children|boy|girl|pedestrians?|crowd|group|someone|figure)\b"),
    "vehicle": (["car", "truck", "bus"], r"\b(cars?|vans?|trucks?|bus|buses|suvs?|sedans?|taxis?|vehicles?|pickups?|jeeps?|minivans?)\b"),
    "two-wheeler": (["bicycle", "motorcycle"], r"\b(bicycles?|bikes?|cyclists?|motorcycles?|motorbikes?|scooters?|motorcyclists?)\b"),
    "bag": (["backpack", "handbag", "suitcase"], r"\b(bags?|backpacks?|handbags?|purses?|suitcases?|luggage|briefcases?|satchels?)\b"),
    "box": (["box", "package"], r"\b(box|boxes|packages?|parcels?|cartons?)\b"),
    "cart": (["shopping cart"], r"\b(carts?|trolleys?)\b"),
    "dog": (["dog"], r"\b(dogs?|pupp(y|ies))\b"),
}
NEGATION = re.compile(r"\b(no|without|not any|none of the)\s+(\w+\s+){0,2}$")


# ---------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------
def video_dirs(out_root) -> list[Path]:
    return sorted(p.parent.parent for p in Path(out_root).glob("*/*/captions/clip_captions.jsonl"))


def load_clips(out_root) -> pd.DataFrame:
    """Every clip caption of every video, one row each (+ vdir = the video's folder)."""
    rows = []
    for vd in video_dirs(out_root):
        for line in (vd / "captions" / "clip_captions.jsonl").read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            r["vdir"] = str(vd)
            rows.append(r)
    return pd.DataFrame(rows)


def _flat(x) -> str:
    if isinstance(x, dict):
        return " ".join(_flat(v) for v in x.values())
    if isinstance(x, list):
        return "; ".join(_flat(v) for v in x)
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def caption_text(r, fields=("summary", "actions", "people", "vehicles", "objects", "notable")) -> str:
    return " | ".join(_flat(r.get(f)) for f in fields if _flat(r.get(f)).strip())


# ---------------------------------------------------------------------------------------------------
# 1. Detector cross-check
# ---------------------------------------------------------------------------------------------------
def mentioned_categories(text: str) -> set[str]:
    """Detector categories a caption mentions (skips 'no vehicles', 'without a bag')."""
    found = set()
    low = text.lower()
    for cat, (_, pat) in CATEGORIES.items():
        for m in re.finditer(pat, low):
            if not NEGATION.search(low[max(0, m.start() - 30):m.start()]):
                found.add(cat)
                break
    return found


_TRACK_CACHE: dict = {}


def _tracks(ev: Path):
    """events/tracks.csv (confirmed tracks) and tracks_raw.csv (per-frame boxes), read once per video."""
    key = str(ev)
    if key not in _TRACK_CACHE:
        tr = pd.read_csv(ev / "tracks.csv") if (ev / "tracks.csv").exists() else None
        rw = pd.read_csv(ev / "tracks_raw.csv", usecols=["track_id", "timestamp_s", "cls"]) \
            if (ev / "tracks_raw.csv").exists() else None
        _TRACK_CACHE[key] = (tr, rw)
    return _TRACK_CACHE[key]


def detected_in_clip(vdir: Path, start_s: float, end_s: float, min_overlap_s: float = 1.0):
    """Categories with a confirmed track overlapping the clip by >= min_overlap_s, and the most people
    tracked in one frame (confirmed tracks only)."""
    ev = Path(vdir) / "events"
    tr, rw_all = _tracks(ev)
    if tr is None:
        return None, None
    ov = (np.minimum(tr.end_s, end_s) - np.maximum(tr.start_s, start_s))
    tr = tr[ov >= min(min_overlap_s, 0.5 * (end_s - start_s))]
    cls2cat = {c: cat for cat, (classes, _) in CATEGORIES.items() for c in classes}
    cats = {cls2cat[c] for c in tr.cls if c in cls2cat}
    n_people = 0
    if rw_all is not None and (tr.cls == "person").any():
        rw = rw_all
        rw = rw[(rw.cls == "person") & rw.track_id.isin(tr.track_id) &
                (rw.timestamp_s >= start_s) & (rw.timestamp_s < end_s)]
        n_people = int(rw.groupby("timestamp_s").size().max()) if len(rw) else 0
    return cats, n_people


def detector_crosscheck(out_root) -> pd.DataFrame:
    """One row per clip: what the caption mentions vs what the detector tracked in the same seconds."""
    _TRACK_CACHE.clear()
    rows = []
    for _, r in load_clips(out_root).iterrows():
        det, n_det = detected_in_clip(Path(r.vdir), r.start_s, r.end_s)
        if det is None:
            continue
        men = mentioned_categories(caption_text(r))
        if (r.get("people_count_max") or 0) > 0:
            men.add("person")
        n_vlm = r.get("people_count_max")
        n_vlm = None if n_vlm is None or (isinstance(n_vlm, float) and np.isnan(n_vlm)) else int(n_vlm)
        rows.append({
            "dataset": r.dataset, "video_id": r.video_id, "clip_id": r.clip_id,
            "time": f"{r.start_hms[3:8]}-{r.end_hms[3:8]}",
            "mentioned": ", ".join(sorted(men)), "detected": ", ".join(sorted(det)),
            "likely_hallucination": ", ".join(sorted(men - det)), "likely_miss": ", ".join(sorted(det - men)),
            "people_vlm": n_vlm, "people_detector": n_det,
            "people_count_ok": None if n_vlm is None else abs(n_vlm - n_det) <= max(1, 0.25 * max(n_vlm, n_det)),
            "summary": r.summary, "vdir": r.vdir})
    return pd.DataFrame(rows)


def crosscheck_summary(cc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per dataset: % clips with a likely hallucination / miss, people-count agreement. Per category: how
    often a mention is backed by the detector (precision-like) and a detection is mentioned (recall-like)."""
    if cc.empty:
        return pd.DataFrame(), pd.DataFrame()
    g = cc.groupby("dataset")
    per_ds = pd.DataFrame({
        "clips": g.size(),
        "clips with a likely hallucination %": g.likely_hallucination.apply(lambda s: 100 * (s != "").mean()),
        "clips with a likely miss %": g.likely_miss.apply(lambda s: 100 * (s != "").mean()),
        "people count within ±25% of detector %": g.people_count_ok.apply(
            lambda s: 100 * s.dropna().astype(bool).mean() if s.notna().any() else np.nan),
        "mean |people VLM - detector|": g.apply(lambda d: (d.people_vlm - d.people_detector).abs().mean()),
    })
    rows = []
    for cat in CATEGORIES:
        men = cc.mentioned.str.contains(rf"\b{cat}\b")
        det = cc.detected.str.contains(rf"\b{cat}\b")
        rows.append({"category": cat, "clips mentioning": int(men.sum()), "clips with a detection": int(det.sum()),
                     "mentions backed by detector %": 100 * (men & det).sum() / men.sum() if men.sum() else np.nan,
                     "detections mentioned %": 100 * (men & det).sum() / det.sum() if det.sum() else np.nan})
    return per_ds.round(1), pd.DataFrame(rows).set_index("category").round(1)


# ---------------------------------------------------------------------------------------------------
# 2. Captions vs ground-truth labels
# ---------------------------------------------------------------------------------------------------
class TextEmbedder:
    """Sentence embeddings (unit length) with a BERT-style model (default bge-small, CLS pooling). If the model
    can't be loaded (no internet, no transformers), falls back to TF-IDF vectors built from the texts given, so
    the numbers still come out (mark them as TF-IDF in the report)."""

    def __init__(self, model: str = TEXT_MODEL, device: str = "auto"):
        self.name, self.kind = model, "tfidf"
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self.torch = torch
            self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
            self.tok = AutoTokenizer.from_pretrained(model)
            self.model = AutoModel.from_pretrained(model).to(self.device).eval()
            self.kind = "model"
        except Exception as e:  # offline / missing package: still give numbers
            print(f"[caption_eval] text model {model} not available ({type(e).__name__}); using TF-IDF instead")

    def encode(self, texts: list[str], batch: int = 64) -> np.ndarray:
        texts = [t if t.strip() else "(empty)" for t in texts]
        if self.kind == "model":
            out = []
            with self.torch.inference_mode():
                for i in range(0, len(texts), batch):
                    b = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=512,
                                 return_tensors="pt").to(self.device)
                    v = self.model(**b).last_hidden_state[:, 0]
                    out.append(self.torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
            return np.concatenate(out)
        return self._tfidf(texts)

    def fit(self, texts: list[str]) -> "TextEmbedder":
        """TF-IDF fallback only: fix the vocabulary and IDF on these texts (e.g. the captions), so later
        encode() calls (e.g. queries) land in the same space. No effect with a sentence model."""
        if self.kind == "tfidf":
            tokens = [re.findall(r"[a-z]+", t.lower()) for t in texts]
            self.vocab = {w: i for i, w in enumerate(sorted({w for ts in tokens for w in ts}))}
            df = np.zeros(max(1, len(self.vocab)), np.float32)
            for ts in tokens:
                for w in set(ts):
                    df[self.vocab[w]] += 1
            self.idf = np.log((1 + len(texts)) / (1 + df)) + 1
        return self

    def _tfidf(self, texts):
        tokens = [re.findall(r"[a-z]+", t.lower()) for t in texts]
        vocab, idf = getattr(self, "vocab", None), getattr(self, "idf", None)
        if vocab is None:   # not fitted: vocabulary from these texts (only comparable within this call)
            vocab = {w: i for i, w in enumerate(sorted({w for ts in tokens for w in ts}))}
        M = np.zeros((len(texts), max(1, len(vocab))), np.float32)
        for i, ts in enumerate(tokens):
            for w in ts:
                if w in vocab:
                    M[i, vocab[w]] += 1
        if idf is None:
            idf = np.log((1 + len(texts)) / (1 + (M > 0).sum(0))) + 1
        M *= idf
        return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)


def _label_text(label: str) -> str:
    return str(label).replace("_", " ")   # MEVA "person_opens_facility_door" -> "person opens facility door"


def label_similarity(out_root, embedder: TextEmbedder | None = None, datasets=("uca", "meva")):
    """Each ground-truth label vs the caption of the clip covering it (most overlap). Returns (pairs, summary).
    Baseline = the same label vs captions of clips from *other* videos. top-1 / top-5 = the label's most similar
    captions (over all clips of the dataset) include a clip covering it."""
    clips = load_clips(out_root)
    if clips.empty:
        return pd.DataFrame(), pd.DataFrame()
    clips = clips[clips.dataset.isin(datasets)].reset_index(drop=True)
    labels = []
    for vd, g in clips.groupby("vdir"):
        gt_path = Path(vd) / "gt.csv"
        if not gt_path.exists():
            continue
        gt = pd.read_csv(gt_path)
        for x in gt.itertuples():
            labels.append({"dataset": g.dataset.iat[0], "video_id": g.video_id.iat[0], "vdir": vd,
                           "start_s": float(x.start_s), "end_s": float(x.end_s), "label": str(x.label)})
    lab = pd.DataFrame(labels)
    if lab.empty:
        return pd.DataFrame(), pd.DataFrame()
    emb = embedder or TextEmbedder()
    cap_txt = [caption_text(r, ("summary", "actions")) for _, r in clips.iterrows()]
    V = emb.encode(cap_txt + [_label_text(s) for s in lab.label])   # together: TF-IDF needs one vocabulary
    C, L = V[:len(cap_txt)], V[len(cap_txt):]
    S = L @ C.T   # label x clip similarity
    rows = []
    for i, x in lab.iterrows():
        same_ds = (clips.dataset == x.dataset).to_numpy()
        in_vid = (clips.vdir == x.vdir).to_numpy()
        ov = (np.minimum(clips.end_s, x.end_s) - np.maximum(clips.start_s, x.start_s)).to_numpy()
        covering = in_vid & (ov > 0)
        r = {"dataset": x.dataset, "video_id": x.video_id, "label": x.label,
             "label_time": f"{x.start_s:.1f}-{x.end_s:.1f}", "covered_by_a_clip": bool(covering.any())}
        others = same_ds & ~in_vid
        r["sim_other_videos"] = float(S[i, others].mean()) if others.any() else np.nan
        if covering.any():
            j = int(np.flatnonzero(covering)[np.argmax(ov[covering])])
            r.update(clip_id=clips.clip_id[j], caption=clips.summary[j], sim_matched=float(S[i, j]))
            order = np.flatnonzero(same_ds)[np.argsort(-S[i, same_ds])]
            r["top1"] = bool(covering[order[:1]].any())
            r["top5"] = bool(covering[order[:5]].any())
        rows.append(r)
    pairs = pd.DataFrame(rows)
    g = pairs.groupby("dataset")
    cov = pairs[pairs.covered_by_a_clip]
    gc = cov.groupby("dataset")
    summary = pd.DataFrame({
        "labels": g.size(),
        "covered by a clip %": g.covered_by_a_clip.mean() * 100,
        "similarity, matching caption": gc.sim_matched.mean(),
        "similarity, other videos' captions": gc.sim_other_videos.mean(),
        "gap": gc.sim_matched.mean() - gc.sim_other_videos.mean(),
        "matching caption beats other videos' average %": gc.apply(lambda d: 100 * (d.sim_matched > d.sim_other_videos).mean()),
        "top-1 is a covering clip %": gc.top1.mean() * 100,
        "top-5 has a covering clip %": gc.top5.mean() * 100,
    }).round(3)
    summary.attrs["embedder"] = f"{emb.name} ({'sentence model' if emb.kind == 'model' else 'TF-IDF fallback'})"
    return pairs, summary


# ---------------------------------------------------------------------------------------------------
# 3. Hand check
# ---------------------------------------------------------------------------------------------------
FILL_COLS = ["reviewer", "hallucinations (count)", "hallucinated (what)", "misses (count)", "missed (what)",
             "people count ok (y/n)", "notes"]


def make_review_sheets(out_root, n: int = 30, reviewers: int = 5, seed: int = 0, overwrite: bool = False) -> Path | None:
    """n random clips (spread over datasets in proportion to their clips) split over `reviewers` people:
    out_root/clip_review/review_<k>.html (frames next to the caption) + review_<k>.csv (fill in).
    Never overwrites sheets that exist unless overwrite=True, so filled-in answers are safe."""
    out_root = Path(out_root)
    folder = out_root / "clip_review"
    if folder.exists() and any(folder.glob("review_*.csv")) and not overwrite:
        return folder
    clips = load_clips(out_root)
    if clips.empty:
        return None
    rng = np.random.default_rng(seed)
    pick = clips.loc[sorted(rng.choice(len(clips), size=min(n, len(clips)), replace=False))].reset_index(drop=True)
    pick = pick.sort_values(["dataset", "video_id", "start_s"]).reset_index(drop=True)
    pick["review_id"] = np.arange(1, len(pick) + 1)
    pick["sheet"] = (pick.index % reviewers) + 1
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("review_*"):
        old.unlink()
    esc = lambda x: str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
    for k, sh in pick.groupby("sheet"):
        rows = []
        for r in sh.itertuples():
            rows.append({"review_id": r.review_id, "dataset": r.dataset, "video_id": r.video_id,
                         "clip_id": f"c{r.clip_id}",   # "c" so Excel doesn't turn "001.0" into 1
                         "time": f"{r.start_hms[3:8]}-{r.end_hms[3:8]}", "summary": r.summary,
                         "actions": _flat(r.actions), "people": _flat(r.people),
                         "people_count_max": r.people_count_max, "vehicles": _flat(r.vehicles),
                         "objects": _flat(r.objects), "notable": r.notable, **{c: "" for c in FILL_COLS}})
        pd.DataFrame(rows).to_csv(folder / f"review_{k}.csv", index=False)
        blocks = []
        for r in sh.itertuples():
            imgs = "".join(f"<figure><img src='{esc(os.path.relpath(Path(r.vdir) / im, folder))}'>"
                           f"<figcaption>+{t - r.start_s:.1f} s</figcaption></figure>"
                           for im, t in zip(r.images, r.frame_times_s))
            blocks.append(
                f"<section><h3>#{r.review_id} · {esc(r.dataset)}/{esc(r.video_id)} · clip {esc(r.clip_id)} · "
                f"{esc(r.start_hms[:8])}-{esc(r.end_hms[:8])}</h3><div class='strip'>{imgs}</div>"
                f"<p><b>Summary:</b> {esc(r.summary)}</p><p><b>Actions:</b> {esc(_flat(r.actions))}</p>"
                f"<p><b>People</b> (max {esc(r.people_count_max)}): {esc(_flat(r.people))}</p>"
                f"<p><b>Vehicles:</b> {esc(_flat(r.vehicles)) or '—'} · <b>Objects:</b> {esc(_flat(r.objects)) or '—'}</p>"
                + (f"<p><b>Notable:</b> {esc(r.notable)}</p>" if r.notable else "") + "</section>")
        html = (
            "<!DOCTYPE html><meta charset='utf-8'><title>Clip review " + str(k) + "</title><style>"
            "body{font-family:sans-serif;margin:16px;max-width:1500px}section{border-bottom:2px solid #ccc;padding:8px 0}"
            ".strip{display:flex;gap:4px;flex-wrap:wrap}figure{margin:0}img{width:240px}figcaption{font-size:12px}"
            "p{margin:4px 0}</style>"
            f"<h2>Clip review sheet {k} ({len(sh)} clips)</h2><p>For each clip, look at the frames and fill in "
            f"<code>review_{k}.csv</code>: <b>hallucinations (count)</b> = things the caption names that are not in the "
            "frames (a person, colour, object, action), <b>misses (count)</b> = clearly visible people / vehicles / "
            "objects / actions it doesn't mention, <b>people count ok</b> = y if the max people count is about right. "
            "Write 0 when there are none. About 1 minute per clip.</p>" + "".join(blocks))
        (folder / f"review_{k}.html").write_text(html)
    return folder


def review_results(out_root, crosscheck: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filled-in sheets -> per dataset: clips reviewed, % with >= 1 hallucination / miss, mean per clip, people
    count ok %. With `crosscheck`, also how often the detector check agreed with the reviewers."""
    folder = Path(out_root) / "clip_review"
    sheets = sorted(folder.glob("review_*.csv")) if folder.exists() else []
    if not sheets:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.concat([pd.read_csv(p, keep_default_na=False, dtype={"clip_id": str, "video_id": str})
                    for p in sheets], ignore_index=True)
    df["clip_id"] = df.clip_id.str.lstrip("c")   # sheets write "c001.0"
    num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
    df["n_hal"], df["n_miss"] = num(df["hallucinations (count)"]), num(df["misses (count)"])
    done = df[df.n_hal.notna()].copy()
    if done.empty:
        return pd.DataFrame(), done
    g = done.groupby("dataset")
    res = pd.DataFrame({
        "clips reviewed": g.size(),
        "clips with >= 1 hallucination %": g.n_hal.apply(lambda s: 100 * (s > 0).mean()),
        "hallucinations per clip": g.n_hal.mean(),
        "clips with >= 1 miss %": g.n_miss.apply(lambda s: 100 * (s > 0).mean()),
        "misses per clip": g.n_miss.mean(),
        "people count ok %": g["people count ok (y/n)"].apply(
            lambda s: 100 * s.astype(str).str.lower().str.startswith("y").mean()),
    })
    if crosscheck is not None and len(crosscheck):
        m = done.merge(crosscheck[["dataset", "video_id", "clip_id", "likely_hallucination", "likely_miss"]],
                       on=["dataset", "video_id", "clip_id"], how="left")
        m["det_hal"], m["det_miss"] = m.likely_hallucination.fillna("") != "", m.likely_miss.fillna("") != ""
        gm = m.groupby("dataset")
        res["detector check agrees on hallucination (y/n) %"] = gm.apply(lambda d: 100 * ((d.n_hal > 0) == d.det_hal).mean())
        res["detector check agrees on miss (y/n) %"] = gm.apply(lambda d: 100 * ((d.n_miss > 0) == d.det_miss).mean())
    return res.round(1), done
