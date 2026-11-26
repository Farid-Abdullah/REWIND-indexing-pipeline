"""Stage 5: VLM captioning. Turn the selected frames of each video into structured text for search.

Run:  python captions.py                          every video under outputs/batch/ with stage 1-2 results
      python captions.py outputs/batch/meva/<video>   one video folder
      python captions.py --mock                   fake captioner, plumbing check only (no GPU, no model)
      python captions.py --review 60              only (re)write the hand-check sheet for caption quality
Also runs as the "vlm" phase of batch.py / pipeline.ipynb.

Two levels, both written to <video folder>/captions/:

  frames  one caption per selected frame. Which frames: stage 4's hand-off (events/index_records.jsonl,
          so a caption and the SigLIP embedding of the same frame share `embedding_row`), or stage 2's
          keyframes.csv when stage 4 hasn't run. Answer = a JSON object: caption, scene, lighting,
          people (appearance / action / where), vehicles, objects, notable, uncertain.
  clips   one caption per clip = a motion segment cut into pieces of at most `clip_max_s`. The VLM sees
          up to `clip_max_frames` frames of the clip in time order, so it can describe what *happens*
          (enters, picks up, leaves), which one frame can't show. A clip has a start and an end: it is
          the unit a search returns and a player jumps to.

Every record also gets `search_text` (all the fields joined into one string) for a keyword or text-
embedding search. The model is told to describe only what it can see and to put doubts in `uncertain`,
because results are used as evidence and must not invent things; caption_review.csv lets you check that
by hand (supervisor's "how often does the VLM mention something that isn't there").

Resume: records already captioned with the same settings (model, prompts, image size) are kept; change a
setting and that video is captioned again. Model weights: any Qwen2.5-VL / Qwen3-VL checkpoint, by Hugging
Face name or local folder (`model`). Needs torch + transformers >= 4.51 (Qwen3-VL: >= 4.57) + accelerate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from video_utils import format_hms, read_frames, video_info

ROOT = Path(__file__).resolve().parent

CAPTION_CONFIG = {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",   # HF name or local folder; Qwen3-VL works too (e.g. Qwen3-VL-8B-Instruct)
    "device": "auto",              # "auto" (cuda if available) | "cuda" | "cuda:1" | "cpu"
    "dtype": "bfloat16",           # L4 supports bf16; 7B in bf16 ~ 16 GB of the 24 GB
    "levels": ["frames", "clips"],
    "source": "auto",              # frames to caption: "index" (stage 4) | "keyframes" (stage 2) | "auto" = index if present
    "frame_max_pixels": 1280 * 28 * 28,   # ~1.0 MP per frame (~1,280 visual tokens for Qwen2.5-VL); small far people need detail
    "clip_max_pixels": 448 * 28 * 28,     # per frame when several frames go in one prompt
    "clip_max_frames": 6,          # frames per clip prompt (stage 2/4 frames first, evenly spaced frames to fill up)
    "clip_min_frames": 3,          # fewer selected frames than this in a clip -> add evenly spaced ones from the video
    "clip_max_s": 60.0,            # longer segments are cut into equal clips of at most this length
    "batch_size": 4,               # frame prompts per generate() call
    "max_new_tokens_frame": 400,
    "max_new_tokens_clip": 500,
    "save_jpg_quality": 90,        # model-input frames saved to captions/frames/ (review sheet, dashboard previews)
    "mock": False,                 # fake captioner: plumbing check only
}

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You describe frames from fixed security cameras for a search index whose results are used as evidence. "
    "Describe only what is clearly visible. Never guess names, identities, intentions or events that are not "
    "shown. If something is too small, dark or blurry to tell, say so in \"uncertain\" instead of guessing. "
    "Answer with one JSON object and nothing else."
)

FRAME_PROMPT = """This is one frame from a fixed security camera.
Return one JSON object with exactly these keys:
{
  "caption": "1-2 plain sentences: who or what is where, doing what",
  "scene": "the place in a few words, e.g. parking lot, school corridor, shop counter",
  "lighting": "day, night, indoor or unclear",
  "people_count": <number of people visible, 0 if none>,
  "people": [{"appearance": "clothing colours, bags or items carried", "action": "walking, standing, running, sitting, lying, ...", "where": "part of the frame and nearby landmark"}],
  "vehicles": [{"type": "car, van, truck, bus, bicycle, motorcycle, ...", "colour": "...", "action": "parked, moving, stopping, ..."}],
  "objects": ["notable objects, e.g. a bag on the floor, a box, a shopping cart"],
  "notable": "anything unusual (fight, fall, person on the ground, fire, smoke, abandoned bag, damage), or empty string",
  "uncertain": "what is too small, dark or blurry to tell, or empty string"
}
List at most 6 people and 6 vehicles (the most visible ones). Use [] for empty lists."""

CLIP_PROMPT = """These {n} frames come from one fixed security camera, in time order, covering {dur:.0f} seconds.
Describe what happens across them. Return one JSON object with exactly these keys:
{{
  "summary": "2-3 plain sentences describing what happens, in order",
  "actions": ["short phrases, e.g. 'person in a red jacket enters through the door', 'white car parks'"],
  "people_count_max": <largest number of people visible in any one frame>,
  "people": ["short appearance descriptions, e.g. 'man in black jacket with a backpack'"],
  "vehicles": ["e.g. 'white van, parked'"],
  "objects": ["notable objects"],
  "notable": "anything unusual (fight, fall, person on the ground, fire, smoke, abandoned bag, damage), or empty string",
  "uncertain": "what cannot be told from these frames, or empty string"
}}
Only describe changes you can see between the frames. Use [] for empty lists."""

FRAME_KEYS = {"caption": "", "scene": "", "lighting": "", "people_count": None, "people": [], "vehicles": [],
              "objects": [], "notable": "", "uncertain": ""}
CLIP_KEYS = {"summary": "", "actions": [], "people_count_max": None, "people": [], "vehicles": [], "objects": [],
             "notable": "", "uncertain": ""}


def log(msg: str) -> None:
    print(f"[captions] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------
class QwenVL:
    """Qwen2.5-VL / Qwen3-VL through Hugging Face transformers. Greedy decoding (same input, same caption)."""

    def __init__(self, model: str, device: str = "auto", dtype: str = "bfloat16"):
        import torch
        import transformers
        from packaging import version
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as AutoVLM
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dt = getattr(torch, dtype) if device != "cpu" else torch.float32
        # `dtype=` replaced `torch_dtype=` in 4.56; an unknown keyword would be ignored and load in fp32 (OOM on an L4)
        kw = {"dtype": dt} if version.parse(transformers.__version__) >= version.parse("4.56") else {"torch_dtype": dt}
        self.model = AutoVLM.from_pretrained(model, **kw).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model)
        self.proc.tokenizer.padding_side = "left"   # batched generation: pad on the left so all answers start together
        self.name = model

    def generate(self, prompts: list[tuple[list, list[Image.Image]]], max_new_tokens: int) -> list[str]:
        """prompts: [(chat messages, the images they reference in order), ...] -> one answer string each."""
        texts = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m, _ in prompts]
        images = [im for _, ims in prompts for im in ims]
        inputs = self.proc(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.proc.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def peak_vram_gb(self) -> float | None:
        if self.device.startswith("cuda"):
            return round(self.torch.cuda.max_memory_allocated() / 1e9, 2)
        return None


class MockVL:
    """Fake captioner for plumbing checks: answers valid JSON built from image brightness. Meaningless text."""

    name = "mock"

    def generate(self, prompts, max_new_tokens):
        outs = []
        for messages, ims in prompts:
            lum = [float(np.asarray(im.convert("L")).mean()) for im in ims]
            light = "day" if np.mean(lum) > 90 else "night"
            if len(ims) == 1:
                outs.append(json.dumps({
                    "caption": f"Mock caption: a scene with mean brightness {lum[0]:.0f}.", "scene": "mock scene",
                    "lighting": light, "people_count": int(lum[0]) % 3,
                    "people": [{"appearance": "grey clothes", "action": "walking", "where": "centre"}],
                    "vehicles": [], "objects": [], "notable": "", "uncertain": "mock"}))
            else:
                outs.append("```json\n" + json.dumps({   # fenced on purpose: exercises the parser
                    "summary": f"Mock clip of {len(ims)} frames, brightness {lum[0]:.0f} to {lum[-1]:.0f}.",
                    "actions": ["person walks across"], "people_count_max": 1, "people": ["person in grey"],
                    "vehicles": [], "objects": [], "notable": "", "uncertain": "mock"}) + "\n```")
        return outs

    def peak_vram_gb(self):
        return None


class Captioner:
    """Loads the model on first use (so a batch that only needs resume checks never loads 16 GB)."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = {**CAPTION_CONFIG, **(cfg or {})}
        self._vl = None
        self.load_s = 0.0

    @property
    def vl(self):
        if self._vl is None:
            t0 = time.perf_counter()
            c = self.cfg
            self._vl = MockVL() if c["mock"] else QwenVL(c["model"], c["device"], c["dtype"])
            self.load_s = time.perf_counter() - t0
            log(f"loaded {self._vl.name} in {self.load_s:.0f} s")
        return self._vl

    @property
    def model_name(self) -> str:
        return "mock" if self.cfg["mock"] else str(self.cfg["model"])

    def settings_hash(self) -> str:
        """Changes when anything that changes a caption changes (model, prompts, image size, clip cutting)."""
        c = self.cfg
        keys = [self.model_name, PROMPT_VERSION, SYSTEM_PROMPT, FRAME_PROMPT, CLIP_PROMPT, c["frame_max_pixels"],
                c["clip_max_pixels"], c["clip_max_frames"], c["clip_min_frames"], c["clip_max_s"],
                c["max_new_tokens_frame"], c["max_new_tokens_clip"]]
        return hashlib.md5(json.dumps(keys, default=str).encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------------------------------
# Prompts and answers
# ---------------------------------------------------------------------------------------------------
def fit_pixels(im: Image.Image, max_pixels: int) -> Image.Image:
    """Shrink (never enlarge) so width x height <= max_pixels, keeping the aspect ratio."""
    w, h = im.size
    if w * h <= max_pixels:
        return im
    s = math.sqrt(max_pixels / (w * h))
    return im.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)


def frame_messages() -> list:
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": FRAME_PROMPT}]}]


def clip_messages(rel_times: list[float], dur: float) -> list:
    content = []
    for i, t in enumerate(rel_times, 1):
        content += [{"type": "text", "text": f"Frame {i} (+{t:.1f} s):"}, {"type": "image"}]
    content.append({"type": "text", "text": CLIP_PROMPT.format(n=len(rel_times), dur=dur)})
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content}]


def parse_answer(text: str) -> dict | None:
    """The model's answer -> dict, or None. Tolerates ```json fences, text around the object, trailing commas."""
    t = re.sub(r"```(?:json)?", "", text or "").strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    s = t[a:b + 1]
    for cand in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except ValueError:
            continue
    return None


def _as_int(x):
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return None


def normalise(obj: dict | None, keys: dict, raw: str) -> dict:
    """Missing keys get defaults, wrong types are fixed. A failed parse keeps the raw text as the caption,
    so the frame stays searchable."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in keys.items()}
    if obj is None:
        main = "caption" if "caption" in keys else "summary"
        out[main] = re.sub(r"\s+", " ", raw or "").strip()[:600]
        return out
    for k, default in keys.items():
        v = obj.get(k, default)
        if isinstance(default, list):
            v = v if isinstance(v, list) else ([] if v in (None, "") else [v])
        elif k.startswith("people_count"):
            v = _as_int(v)
        else:
            v = "" if v is None else (v if isinstance(v, str) else json.dumps(v))
        out[k] = v
    return out


def _flat(x) -> str:
    if isinstance(x, dict):
        return " ".join(_flat(v) for v in x.values())
    if isinstance(x, list):
        return "; ".join(_flat(v) for v in x)
    return "" if x is None else str(x)


def search_text(fields: dict) -> str:
    """One string with everything a text search should match (people, vehicles, objects, actions...)."""
    order = ["caption", "summary", "actions", "scene", "lighting", "people", "vehicles", "objects", "notable"]
    parts = [_flat(fields.get(k)) for k in order if fields.get(k)]
    return re.sub(r"\s+", " ", " | ".join(p for p in parts if p.strip())).strip()


# ---------------------------------------------------------------------------------------------------
# What to caption
# ---------------------------------------------------------------------------------------------------
def load_frames_to_caption(vdir: Path, source: str = "auto") -> tuple[pd.DataFrame, str]:
    """The frames of one video to caption, from stage 4 (index records) or stage 2 (keyframes)."""
    rec = vdir / "events" / "index_records.jsonl"
    if source in ("auto", "index") and rec.exists() and rec.stat().st_size > 0:
        df = pd.read_json(rec, lines=True)
        df["reason"] = "index"
        return df.sort_values("timestamp_s").reset_index(drop=True), "index"
    if source == "index":
        raise FileNotFoundError(f"{rec} missing or empty (run the GPU phase, or use source='keyframes')")
    kf = vdir / "keyframes.csv"
    if not kf.exists():
        raise FileNotFoundError(f"{kf} missing: run stages 1-2 first")
    df = pd.read_csv(kf)
    return df.sort_values("timestamp_s").reset_index(drop=True), "keyframes"


def make_clips(segments: pd.DataFrame, max_s: float) -> pd.DataFrame:
    """Each motion segment, cut into equal pieces of at most max_s seconds."""
    rows = []
    for seg in segments.itertuples(index=False):
        n = max(1, math.ceil((seg.end_s - seg.start_s) / max_s - 1e-9))
        edges = np.linspace(seg.start_s, seg.end_s, n + 1)
        for k in range(n):
            rows.append({"clip_id": f"{int(seg.segment_id):03d}.{k}", "segment_id": int(seg.segment_id),
                         "start_s": round(float(edges[k]), 3), "end_s": round(float(edges[k + 1]), 3)})
    return pd.DataFrame(rows, columns=["clip_id", "segment_id", "start_s", "end_s"])


def clip_frame_indices(clip, frames: pd.DataFrame, fps: float, n_frames: int, max_frames: int,
                       min_frames: int) -> list[int]:
    """Frame numbers shown to the VLM for one clip: the selected frames inside it (evenly thinned to
    max_frames), topped up with evenly spaced frames when fewer than min_frames were selected."""
    inside = frames[(frames.timestamp_s >= clip.start_s) & (frames.timestamp_s < clip.end_s)]
    idx = sorted(set(int(i) for i in inside.frame_idx))
    if len(idx) < min_frames:
        ts = np.linspace(clip.start_s, clip.end_s, max_frames + 2)[1:-1]
        idx = sorted(set(idx) | {min(n_frames - 1, int(round(t * fps))) for t in ts})
    if len(idx) > max_frames:
        idx = [idx[i] for i in sorted(set(np.linspace(0, len(idx) - 1, max_frames).round().astype(int)))]
    return idx


class FrameStore:
    """Model-input frames as JPGs in captions/frames/ (decoded once per video, reused by both levels, the
    review sheet and the dashboard). Saved at frame_max_pixels, so the reviewer sees what the model saw."""

    def __init__(self, vdir: Path, video: Path | None, max_pixels: int, quality: int, fallback_jpgs: dict):
        self.dir = vdir / "captions" / "frames"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.video, self.max_pixels, self.quality, self.fallback = video, max_pixels, quality, fallback_jpgs

    def path(self, frame_idx: int) -> Path:
        return self.dir / f"f{int(frame_idx):07d}.jpg"

    def ensure(self, frame_idxs) -> list[int]:
        """Write the JPGs that don't exist yet; returns the frame numbers that could not be read."""
        missing = sorted({int(i) for i in frame_idxs if not self.path(i).exists()})
        if not missing:
            return []
        got = set()
        if self.video is not None and Path(self.video).exists():
            import cv2
            for fi, bgr in read_frames(self.video, missing):
                self._save(fi, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                got.add(fi)
        for fi in missing:  # video gone or unreadable frame: use stage 2's JPG if there is one
            if fi not in got and fi in self.fallback and Path(self.fallback[fi]).exists():
                self._save(fi, Image.open(self.fallback[fi]).convert("RGB"))
                got.add(fi)
        return [fi for fi in missing if fi not in got]

    def _save(self, fi: int, im: Image.Image) -> None:
        fit_pixels(im, self.max_pixels).save(self.path(fi), quality=self.quality)

    def load(self, frame_idx: int, max_pixels: int | None = None) -> Image.Image:
        im = Image.open(self.path(frame_idx)).convert("RGB")
        return fit_pixels(im, max_pixels) if max_pixels else im


# ---------------------------------------------------------------------------------------------------
# One video
# ---------------------------------------------------------------------------------------------------
def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _load_done(path: Path, key: str, h: str) -> dict:
    """Records already captioned with the same settings, by key (frame_idx or clip_id)."""
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("settings_hash") == h:
                done[r[key]] = r
    return done


def _rewrite(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def _append(path: Path, records: list[dict]) -> None:
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def caption_video(vdir: str | Path, captioner: Captioner, force: bool = False,
                  levels: list[str] | None = None) -> dict:
    """Caption one video folder of the batch (needs summary_cpu.json, segments.csv, keyframes.csv).
    `levels` (default: the captioner's) = which of "clips" / "frames" to caption now; the other level's
    finished records are kept. Writes captions/frame_captions.jsonl, clip_captions.jsonl, captions.csv
    (both levels, everything captioned so far), frames/*.jpg and summary_vlm.json. Returns the summary."""
    vdir = Path(vdir)
    c = captioner.cfg
    levels = list(levels or c["levels"])
    s1 = _read_json(vdir / "summary_cpu.json") or {}
    if s1.get("status_cpu") != "ok":
        raise RuntimeError(f"{vdir}: stage 1-2 results missing (summary_cpu.json); run the CPU phase first")
    video = Path(s1["path"]) if s1.get("path") else None
    info = video_info(video) if video is not None and video.exists() else None
    fps = info.fps if info else float(s1.get("fps") or 30.0)
    n_frames = info.frame_count if info else int(float(s1.get("duration_s", 0)) * fps)
    t_start = datetime.fromisoformat(s1["start_wallclock"]) if s1.get("start_wallclock") else None
    base = {"dataset": s1.get("dataset", ""), "video_id": s1.get("video_id", vdir.name),
            "camera_id": str(s1.get("camera_id", vdir.name)), "video_path": str(video or "")}

    def wall(t):
        return (t_start + timedelta(seconds=float(t))).isoformat() if t_start else None

    cap_dir = vdir / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)
    h = captioner.settings_hash()
    frames, source = load_frames_to_caption(vdir, c["source"])
    segments = pd.read_csv(vdir / "segments.csv")
    clips = make_clips(segments, c["clip_max_s"]) if "clips" in levels else make_clips(segments.iloc[0:0], 1)
    kf_jpgs = {}
    if (vdir / "keyframes.csv").exists():
        kf = pd.read_csv(vdir / "keyframes.csv")
        kf_jpgs = {int(r.frame_idx): vdir / "keyframes" / r.filename for r in kf.itertuples() if isinstance(r.filename, str)}
    store = FrameStore(vdir, video, c["frame_max_pixels"], c["save_jpg_quality"], kf_jpgs)
    clip_idx = {r.clip_id: clip_frame_indices(r, frames, fps, n_frames, c["clip_max_frames"], c["clip_min_frames"])
                for r in clips.itertuples()}

    prev = _read_json(vdir / "summary_vlm.json") or {}
    if prev.get("settings_hash") != h:
        prev = {}   # other settings: earlier speed numbers don't apply
    s = {"status_vlm": "ok", "vlm_model": captioner.model_name, "vlm_source": source, "settings_hash": h,
         "vlm_frames_new": 0, "vlm_clips_new": 0}
    T = {"vlm_decode_s": 0.0, "vlm_frames_s": 0.0, "vlm_clips_s": 0.0}

    # --- decode every frame the requested levels need, once
    need = set(int(i) for i in frames.frame_idx) if "frames" in levels else set()
    for v in clip_idx.values():
        need |= set(v)
    t0 = time.perf_counter()
    bad = set(store.ensure(need))
    T["vlm_decode_s"] = time.perf_counter() - t0
    s["vlm_unreadable_frames"] = len(bad)

    # --- level: frames
    fpath = cap_dir / "frame_captions.jsonl"
    if "frames" in levels:
        done = {} if force else _load_done(fpath, "frame_idx", h)
        todo = [r for r in frames.itertuples() if int(r.frame_idx) not in done and int(r.frame_idx) not in bad]
        _rewrite(fpath, [done[k] for k in sorted(done)])
        t0 = time.perf_counter()
        bs = max(1, int(c["batch_size"]))
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            prompts = [(frame_messages(), [store.load(int(r.frame_idx))]) for r in chunk]
            tg = time.perf_counter()
            answers = captioner.vl.generate(prompts, c["max_new_tokens_frame"])
            dt = (time.perf_counter() - tg) / len(chunk)
            recs = []
            for r, raw in zip(chunk, answers):
                obj = parse_answer(raw)
                fields = normalise(obj, FRAME_KEYS, raw)
                rec = {**base, "level": "frame", "frame_idx": int(r.frame_idx),
                       "timestamp_s": round(float(r.timestamp_s), 3), "timestamp_hms": format_hms(r.timestamp_s),
                       "wallclock": wall(r.timestamp_s), "segment_id": int(r.segment_id),
                       "reason": getattr(r, "reason", source),
                       "embedding_row": int(r.embedding_row) if hasattr(r, "embedding_row") else None,
                       "detector_events": list(getattr(r, "events", []) or []) if source == "index" else [],
                       "detector_objects": dict(getattr(r, "objects", {}) or {}) if source == "index" else {},
                       "image": str(store.path(r.frame_idx).relative_to(vdir)),
                       **fields, "search_text": search_text(fields), "parse_ok": obj is not None,
                       "raw": raw, "gen_s": round(dt, 3), "model": captioner.model_name,
                       "prompt_version": PROMPT_VERSION, "settings_hash": h}
                recs.append(rec)
                done[rec["frame_idx"]] = rec
            _append(fpath, recs)   # written per batch: an interrupted run keeps what it has
            s["vlm_frames_new"] += len(recs)
            if (i // bs) % 10 == 0:
                log(f"{base['video_id']}: frames {min(i + bs, len(todo))}/{len(todo)} ({dt:.1f} s/frame)")
        T["vlm_frames_s"] = time.perf_counter() - t0
        frame_recs = [done[k] for k in sorted(done)]
    else:   # not requested now: keep what an earlier run captioned
        done = _load_done(fpath, "frame_idx", h)
        frame_recs = [done[k] for k in sorted(done)]

    # --- level: clips
    cpath = cap_dir / "clip_captions.jsonl"
    if "clips" in levels:
        done = {} if force else _load_done(cpath, "clip_id", h)
        _rewrite(cpath, [done[k] for k in sorted(done)])
        t0 = time.perf_counter()
        todo = [r for r in clips.itertuples() if r.clip_id not in done]
        for j, r in enumerate(todo):
            idx = [i for i in clip_idx[r.clip_id] if i not in bad]
            if not idx:
                continue
            times = [i / fps for i in idx]
            dur = float(r.end_s - r.start_s)
            prompt = (clip_messages([t - r.start_s for t in times], dur),
                      [store.load(i, c["clip_max_pixels"]) for i in idx])
            tg = time.perf_counter()
            raw = captioner.vl.generate([prompt], c["max_new_tokens_clip"])[0]
            obj = parse_answer(raw)
            fields = normalise(obj, CLIP_KEYS, raw)
            rec = {**base, "level": "clip", "clip_id": r.clip_id, "segment_id": int(r.segment_id),
                   "start_s": float(r.start_s), "end_s": float(r.end_s),
                   "start_hms": format_hms(r.start_s), "end_hms": format_hms(r.end_s),
                   "wallclock_start": wall(r.start_s), "wallclock_end": wall(r.end_s),
                   "frame_idxs": idx, "frame_times_s": [round(t, 3) for t in times],
                   "images": [str(store.path(i).relative_to(vdir)) for i in idx],
                   **fields, "search_text": search_text(fields), "parse_ok": obj is not None,
                   "raw": raw, "gen_s": round(time.perf_counter() - tg, 3), "model": captioner.model_name,
                   "prompt_version": PROMPT_VERSION, "settings_hash": h}
            _append(cpath, [rec])
            done[r.clip_id] = rec
            s["vlm_clips_new"] += 1
            if j % 10 == 0:
                log(f"{base['video_id']}: clips {j + 1}/{len(todo)} ({rec['gen_s']:.1f} s/clip)")
        T["vlm_clips_s"] = time.perf_counter() - t0
        clip_recs = [done[k] for k in sorted(done)]
    else:   # not requested now: keep what an earlier run captioned
        done = _load_done(cpath, "clip_id", h)
        clip_recs = [done[k] for k in sorted(done)]

    # --- one flat table with everything captioned so far: what a search or the dashboard reads
    write_captions_csv(cap_dir / "captions.csv", frame_recs, clip_recs)
    s.update(summarise_vlm(prev, s, T, frame_recs, clip_recs, len(frames), len(clip_idx) if "clips" in levels
                           else prev.get("vlm_clips_total", len(clip_recs)), float(s1.get("duration_s") or 0)))
    s["vlm_levels_run"] = levels
    s["vlm_peak_vram_gb"] = captioner._vl.peak_vram_gb() if captioner._vl is not None else prev.get("vlm_peak_vram_gb")
    (vdir / "summary_vlm.json").write_text(json.dumps(s, indent=1, default=str))
    return s


def summarise_vlm(prev: dict, s: dict, T: dict, frame_recs, clip_recs, n_frames_total, n_clips_total, D) -> dict:
    """Counts, and speeds that survive resumes: seconds per frame / clip are measured on the items captioned
    in this run (or kept from an earlier run with the same settings), and the cost of a full video is
    estimated from them, so an interrupted-and-resumed video still gets an honest minutes-per-hour figure."""
    out = {"vlm_frames": len(frame_recs), "vlm_clips": len(clip_recs),
           "vlm_frames_total": n_frames_total, "vlm_clips_total": n_clips_total,
           "vlm_parse_failed": sum(not r["parse_ok"] for r in frame_recs + clip_recs)}
    for unit, n_new in (("frame", s["vlm_frames_new"]), ("clip", s["vlm_clips_new"])):
        key, t = f"vlm_s_per_{unit}", T[f"vlm_{unit}s_s"]
        out[key] = t / n_new if n_new else prev.get(key)
        out[f"vlm_{unit}s_s"] = t if n_new else prev.get(f"vlm_{unit}s_s", 0.0)   # time of the last run that captioned
    out["vlm_decode_s"] = T["vlm_decode_s"] + (prev.get("vlm_decode_s") or 0.0)
    out["vlm_s"] = (out["vlm_frames_s"] or 0.0) + (out["vlm_clips_s"] or 0.0) + out["vlm_decode_s"]
    est = (n_frames_total * (out["vlm_s_per_frame"] or 0.0) if out["vlm_frames"] else 0.0) + \
          (n_clips_total * (out["vlm_s_per_clip"] or 0.0) if out["vlm_clips"] else 0.0)
    out["vlm_min_per_video_hour"] = est / 60 / (D / 3600) if D > 0 and est > 0 else None
    out["vlm_complete"] = out["vlm_frames"] >= n_frames_total and out["vlm_clips"] >= n_clips_total
    return out


def write_captions_csv(path: Path, frame_recs: list[dict], clip_recs: list[dict]) -> pd.DataFrame:
    """Frames and clips in one table: level, camera, start/end (a frame: start = end = its time), text.
    id = "f<frame number>" or "c<clip id>" (text, so "c000.1" isn't read back as a number)."""
    rows = []
    for r in frame_recs:
        rows.append({"level": "frame", "dataset": r["dataset"], "video_id": r["video_id"], "camera_id": r["camera_id"],
                     "video_path": r.get("video_path", ""),
                     "id": f"f{r['frame_idx']}", "segment_id": r["segment_id"], "start_s": r["timestamp_s"],
                     "end_s": r["timestamp_s"], "start_hms": r["timestamp_hms"], "wallclock": r["wallclock"],
                     "text": r["caption"], "people_count": r["people_count"], "notable": r["notable"],
                     "search_text": r["search_text"], "image": r["image"], "embedding_row": r.get("embedding_row"),
                     "parse_ok": r["parse_ok"]})
    for r in clip_recs:
        rows.append({"level": "clip", "dataset": r["dataset"], "video_id": r["video_id"], "camera_id": r["camera_id"],
                     "video_path": r.get("video_path", ""),
                     "id": f"c{r['clip_id']}", "segment_id": r["segment_id"], "start_s": r["start_s"], "end_s": r["end_s"],
                     "start_hms": r["start_hms"], "wallclock": r["wallclock_start"], "text": r["summary"],
                     "people_count": r["people_count_max"], "notable": r["notable"],
                     "search_text": r["search_text"], "image": r["images"][len(r["images"]) // 2] if r["images"] else "",
                     "embedding_row": None, "parse_ok": r["parse_ok"]})
    df = pd.DataFrame(rows)
    if len(df):
        df["embedding_row"] = df["embedding_row"].astype("Int64")   # stays an integer next to empty clip rows
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------------------------------
# Caption quality: a sheet to check by hand
# ---------------------------------------------------------------------------------------------------
def make_review_sheet(out_root: str | Path, n: int = 60, seed: int = 0) -> Path | None:
    """Random frame captions from every captioned video -> out_root/caption_review.csv (fill in the empty
    columns) + caption_review.html (picture next to caption, to read while filling in the CSV).
    Count per caption: things mentioned that are not in the picture (hallucinations) and things clearly
    visible that were missed."""
    out_root = Path(out_root)
    recs = []
    for p in sorted(out_root.glob("*/*/captions/frame_captions.jsonl")):
        vdir = p.parent.parent
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            r["image_path"] = str((vdir / r["image"]).relative_to(out_root))
            recs.append(r)
    if not recs:
        return None
    rng = np.random.default_rng(seed)
    pick = [recs[i] for i in sorted(rng.choice(len(recs), size=min(n, len(recs)), replace=False))]
    df = pd.DataFrame([{
        "review_id": i, "dataset": r["dataset"], "video_id": r["video_id"], "timestamp_hms": r["timestamp_hms"],
        "image": r["image_path"], "caption": r["caption"], "people_count": r["people_count"],
        "people": _flat(r["people"]), "vehicles": _flat(r["vehicles"]), "objects": _flat(r["objects"]),
        "notable": r["notable"], "uncertain": r["uncertain"], "model": r["model"],
        "hallucinated_items": "", "missed_items": "", "count_correct (y/n)": "", "notes": ""}
        for i, r in enumerate(pick)])
    csv = out_root / "caption_review.csv"
    df.to_csv(csv, index=False)
    esc = lambda x: (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))  # noqa: E731
    rows = "\n".join(
        f"<tr><td>{r.review_id}</td><td><img src='{esc(r.image)}' loading='lazy'></td><td><b>{esc(r.dataset)}/"
        f"{esc(r.video_id)} @ {esc(r.timestamp_hms)}</b><p>{esc(r.caption)}</p><p><i>people ({esc(r.people_count)}):"
        f"</i> {esc(r.people)}</p><p><i>vehicles:</i> {esc(r.vehicles)}</p><p><i>objects:</i> {esc(r.objects)}</p>"
        f"<p><i>notable:</i> {esc(r.notable)}</p><p><i>uncertain:</i> {esc(r.uncertain)}</p></td></tr>"
        for r in df.itertuples())
    html = (f"<!DOCTYPE html><meta charset='utf-8'><title>Caption review</title><style>body{{font-family:sans-serif;"
            f"margin:16px}} td{{vertical-align:top;padding:8px;border-bottom:1px solid #ccc}} img{{max-width:640px}}"
            f" p{{margin:4px 0}}</style><h2>Caption review ({len(df)} frames, model {esc(df.model.iat[0])})</h2>"
            f"<p>Fill in caption_review.csv: hallucinated_items = things named that are not in the picture; "
            f"missed_items = clearly visible people/vehicles/objects not mentioned.</p><table>{rows}</table>")
    (out_root / "caption_review.html").write_text(html)
    return csv


def review_summary(csv: str | Path) -> dict:
    """After filling in caption_review.csv: share of captions with at least one hallucinated item, etc."""
    df = pd.read_csv(csv, keep_default_na=False)
    filled = df[(df.hallucinated_items != "") | (df.missed_items != "") | (df["count_correct (y/n)"] != "")]
    if filled.empty:
        return {"reviewed": 0}
    clean = filled.hallucinated_items.astype(str).str.strip().str.lower().isin(["", "0", "none", "-"])
    return {"reviewed": len(filled), "captions_with_hallucination_pct": 100 * float((~clean).mean()),
            "count_correct_pct": 100 * float(filled["count_correct (y/n)"].astype(str).str.lower().str.startswith("y").mean())}


# ---------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------
def video_dirs(out_root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(out_root).glob("*/*/summary_cpu.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5: VLM captions for the selected frames and clips.")
    ap.add_argument("folders", nargs="*", help="video folders (default: every one under --out-root)")
    ap.add_argument("--out-root", default=str(ROOT / "outputs" / "batch"))
    ap.add_argument("--model", default=CAPTION_CONFIG["model"])
    ap.add_argument("--device", default=CAPTION_CONFIG["device"])
    ap.add_argument("--levels", nargs="+", default=CAPTION_CONFIG["levels"], choices=["frames", "clips"])
    ap.add_argument("--source", default=CAPTION_CONFIG["source"], choices=["auto", "index", "keyframes"])
    ap.add_argument("--batch-size", type=int, default=CAPTION_CONFIG["batch_size"])
    ap.add_argument("--force", action="store_true", help="caption again even if done with the same settings")
    ap.add_argument("--mock", action="store_true", help="fake captioner: plumbing check only")
    ap.add_argument("--review", type=int, default=60, help="frames in caption_review.csv (0 = don't write it)")
    ap.add_argument("--review-only", action="store_true", help="only rewrite the review sheet")
    a = ap.parse_args()

    out_root = Path(a.out_root)
    if not a.review_only:
        cap = Captioner({"model": a.model, "device": a.device, "levels": a.levels, "source": a.source,
                         "batch_size": a.batch_size, "mock": a.mock})
        dirs = [Path(f) for f in a.folders] or video_dirs(out_root)
        log(f"{len(dirs)} video folders, model {cap.model_name}")
        for i, d in enumerate(dirs, 1):
            try:
                s = caption_video(d, cap, force=a.force)
                log(f"[{i}/{len(dirs)}] {d.name}: {s['vlm_frames']} frames, {s['vlm_clips']} clips "
                    f"({s['vlm_frames_new']} + {s['vlm_clips_new']} new, {s['vlm_parse_failed']} unparsed) "
                    f"[{s['vlm_s']:.0f} s]")
            except Exception as e:
                log(f"[{i}/{len(dirs)}] {d.name}: ERROR {e!r}")
    if a.review:
        p = make_review_sheet(out_root, a.review)
        log(f"review sheet: {p}" if p else "no captions found for a review sheet")


if __name__ == "__main__":
    main()
