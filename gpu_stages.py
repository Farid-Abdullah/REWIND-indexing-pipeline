"""GPU stages of the REWIND pipeline (from Muzayyan's v2 pipeline). Used by pipeline.ipynb.

Stage 1 (diff + MOG2 motion gating) and the CPU keyframes live in motion_gating.py and keyframes.py.
This file adds what needs deep models:

  Deep gate   scan_deep: DINOv2 patch features vs a slowly learned background -> score_deep, a third
              motion score next to diff and MOG2 (2 frames/s)
  Stage 3     events, only inside motion segments, 4 frames/s:
                tracks  YOLO-World detector + ByteTrack -> rules: appears, loitering, running,
                        object_left, crowd
                siglip  SigLIP zero-shot frame tags vs an event prompt bank; the same embeddings
                        are the search index
                xclip   X-CLIP zero-shot 8-frame clip tags (sees motion over time)
              merged into one event table + an importance curve
  Stage 4     samplers: which candidate frames go to the captioner / index
                phash (one global hash) | novelty (new in SigLIP space) | kcenter (importance-weighted
                coverage, primary) | importance (along the importance curve), all scored with
                coverage / redundancy / event recall / track recall
  Hand-off    write_index_records: index_records.jsonl + index_embeddings.npy

Extra packages: torch transformers ultralytics lap sentencepiece protobuf. A model that can't load is
skipped with a warning (Models.failed). SigLIP's tokenizer needs sentencepiece.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import warnings
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from video_utils import format_hms, read_frames

# =============================================================================
# Configuration
# =============================================================================

# Event prompt bank shared by SigLIP (E2) and X-CLIP (E3). Track rules (E1)
# emit some of the same keys so cross-method agreement can be measured.
# This bank only *tags* events; open-world search later uses the raw
# embeddings, so anything describable stays findable even if not listed here.
EVENT_BANK = [
    {"key": "person_walking", "siglip": ["a person walking", "people walking through the area"],
     "xclip": "a person walking"},
    {"key": "carrying_object", "siglip": ["a person carrying a box", "a person carrying a bag"],
     "xclip": "a person carrying a box or bag"},
    {"key": "running", "siglip": ["a person running"], "xclip": "a person running"},
    {"key": "fighting", "siglip": ["people fighting", "a person hitting another person"],
     "xclip": "people fighting"},
    {"key": "falling", "siglip": ["a person lying on the ground", "a person falling down"],
     "xclip": "a person falling down"},
    {"key": "crowd", "siglip": ["a crowd of people", "a large group of people"],
     "xclip": "a crowd of people gathering"},
    {"key": "vehicle_moving", "siglip": ["a car driving", "a truck driving"], "xclip": "a car driving"},
    {"key": "vehicle_parking", "siglip": ["a car parking", "a parked car with its door open"],
     "xclip": "a car parking"},
    {"key": "object_left", "siglip": ["an abandoned bag on the floor", "an unattended suitcase"],
     "xclip": "a person leaving a bag behind"},
    {"key": "delivery", "siglip": ["a delivery person with a package"], "xclip": "a person delivering a package"},
    {"key": "door_interaction", "siglip": ["a person opening a door"], "xclip": "a person opening a door"},
    {"key": "cycling", "siglip": ["a person riding a bicycle"], "xclip": "a person riding a bicycle"},
]
BACKGROUND_PROMPTS = {
    "siglip": ["an empty room", "an empty street", "a quiet scene with no people",
               "a surveillance camera view where nothing is happening"],
    "xclip": "nothing is happening",
}

DEFAULT_CONFIG = {
    "deep_gate": {  # DINOv2 motion gate (scan_deep)
        "ignore_polygons": [],      # normalised [(x, y), ...] polygons to ignore
        "deep_fps": 2.0,            # runs slower than stage 1's 5 samples/s; held constant in between
        "deep_model": "facebook/dinov2-small",
        "deep_input_h": 224,        # multiple of 14 (DINOv2 patch); width follows aspect ratio
        "deep_z": 3.0,              # per-patch z-score for "changed"
        "deep_min_dist": 0.05,      # and absolute cosine distance floor
        "deep_alpha_bg": 0.05,      # background feature update rate (static patches)
        "deep_alpha_fg": 0.008,     # slow absorb of stopped objects (~1 min at 2 fps)
        "deep_alpha_stats": 0.02,   # running mean/var of distances
        "deep_warmup": 10,          # deep-gate samples used only to initialise
        "batch_size": 16,
    },
    "segments": {
        "heartbeat_s": 300.0,       # idle frames kept so empty scenes are still indexed
    },
    "events": {
        "event_fps": 4.0,           # frames decoded inside segments for stages 2 + 3
        "frame_width": 640,         # resolution handed to the detector
        "chunk": 128,               # frames processed per chunk (bounds memory)
        "methods": ["tracks", "siglip", "xclip"],
        # E1
        "detector_model": "yolov8s-worldv2.pt",
        "detector_vocab": ["person", "car", "truck", "bus", "bicycle", "motorcycle", "backpack",
                           "handbag", "suitcase", "box", "package", "shopping cart", "dog"],
        "detector_conf": 0.15,
        "detector_imgsz": 640,
        "tracker": "bytetrack.yaml",
        "min_track_s": 1.0,
        "loiter_s": 20.0,
        "loiter_radius": 1.5,       # in box heights
        "run_speed": 1.2,           # box heights per second
        "static_s": 10.0,
        "static_radius": 0.3,
        "crowd_n": 5,
        "crowd_min_s": 2.0,
        "carried_objects": ["backpack", "handbag", "suitcase", "box", "package"],
        "vehicles": ["car", "truck", "bus", "motorcycle"],
        # E2
        "siglip_model": "google/siglip-base-patch16-224",
        "siglip_margin": 1.0,       # logit margin of event prompt over best background prompt
        "siglip_smooth": 3,         # samples
        # E3
        "xclip_model": "microsoft/xclip-base-patch32",
        "xclip_frames": 8,          # frames per clip (model input)
        "xclip_stride": 4,          # in event-pass frames (4 @ 4 fps = 1 s)
        "xclip_threshold": 0.35,
        # merging
        "merge_gap_s": 2.0,
        "batch_size": 32,
    },
    "sampling": {
        "methods": ["phash", "novelty", "kcenter", "importance"],
        "primary": "kcenter",
        "export_jpgs": ["kcenter"],
        "min_gap_s": 1.0,           # applies to phash + novelty
        "max_gap_s": 30.0,
        # S0
        "phash_change": 10,         # Hamming bits (of 64)
        # S1
        "novelty_tau": 0.10,        # cosine distance to the nearest kept frame of the segment
        "novelty_adaptive_q": 0.90, # tau = max(novelty_tau, q-quantile of step distances)
        # S2 / S3 shared budget
        "rate_per_min": 8.0,        # base keyframes per minute of active video
        "min_per_segment": 2,
        "max_per_segment": 40,
        "kcenter_lambda_t": 0.15,   # weight of temporal distance in k-center
        "kcenter_radius": 0.06,     # stop early once every frame is this close (adaptive budget)
        "dedup_sim": 0.97,
        "naive_fps": 1.0,           # baseline for reduction factor
    },
    "runtime": {"device": "auto", "fp16": True, "mock_models": False, "cache": True},
}


def deep_update(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Video helpers
# =============================================================================

def resize_w(img, width):
    h, w = img.shape[:2]
    if w == width:
        return img
    return cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def polygon_mask(shape_hw, polygons):
    h, w = shape_hw
    keep = np.ones((h, w), np.uint8)
    for poly in polygons or []:
        pts = np.array([[x * w, y * h] for x, y in poly], np.int32)
        cv2.fillPoly(keep, [pts], 0)
    return keep.astype(bool)


def phash64(gray) -> np.ndarray:
    """64-bit perceptual hash (DCT of 32x32, top-left 8x8 vs median). Returns 64 bools."""
    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    return d > np.median(d[1:])


def hamming(a, b) -> int:
    return int(np.count_nonzero(a != b))


# =============================================================================
# Model wrappers (thin; everything returns numpy so the algorithms are testable)
# =============================================================================

def pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _as_tensor(out):
    """transformers >=5 may return a ModelOutput from get_*_features; older returns a tensor."""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    return out


class PatchEncoder:
    """DINOv2 patch features without centre crop (keeps the full field of view)."""

    def __init__(self, name, device, fp16, input_h=224):
        import torch
        from transformers import AutoModel
        self.torch, self.device = torch, device
        self.dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        self.model = AutoModel.from_pretrained(name).to(device, self.dtype).eval()
        self.patch = getattr(self.model.config, "patch_size", 14)
        self.n_reg = getattr(self.model.config, "num_register_tokens", 0) or 0
        self.input_h = input_h - input_h % self.patch
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __call__(self, frames_bgr):
        h0, w0 = frames_bgr[0].shape[:2]
        H = self.input_h
        W = max(self.patch, round(w0 * H / h0 / self.patch) * self.patch)
        x = np.stack([(cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - self.mean) / self.std
                      for f in frames_bgr]).transpose(0, 3, 1, 2)
        with self.torch.inference_mode():
            t = self.torch.from_numpy(x).to(self.device, self.dtype)
            tok = self.model(pixel_values=t).last_hidden_state[:, 1 + self.n_reg:, :]
            tok = self.torch.nn.functional.normalize(tok.float(), dim=-1)
        gh, gw = H // self.patch, W // self.patch
        return tok.reshape(len(frames_bgr), gh, gw, -1).cpu().numpy()


class SigLIPEncoder:
    def __init__(self, name, device, fp16):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.device = torch, device
        self.dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        self.model = AutoModel.from_pretrained(name).to(device, self.dtype).eval()
        self.proc = AutoProcessor.from_pretrained(name)
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self.logit_bias = float(self.model.logit_bias.item())

    def images(self, frames_bgr):
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        with self.torch.inference_mode():
            px = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            e = _as_tensor(self.model.get_image_features(pixel_values=px)).float()
            return self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    def texts(self, texts):
        with self.torch.inference_mode():
            tok = self.proc(text=texts, padding="max_length", return_tensors="pt").to(self.device)
            e = _as_tensor(self.model.get_text_features(input_ids=tok["input_ids"])).float()
            return self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


class XCLIPScorer:
    def __init__(self, name, device, fp16):
        import torch
        from transformers import XCLIPModel, XCLIPProcessor
        self.torch, self.device = torch, device
        self.model = XCLIPModel.from_pretrained(name).to(device).eval()
        self.proc = XCLIPProcessor.from_pretrained(name)
        self.num_frames = getattr(self.model.config.vision_config, "num_frames", 8)

    def __call__(self, clips_bgr, texts):
        """clips_bgr: list of clips, each a list of num_frames BGR frames -> probs (n_clips, n_texts)."""
        vids = [[cv2.cvtColor(cv2.resize(f, (224, 224)), cv2.COLOR_BGR2RGB) for f in c] for c in clips_bgr]
        with self.torch.inference_mode():
            inp = self.proc(text=texts, videos=vids, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**inp).logits_per_video
            return logits.softmax(-1).float().cpu().numpy()


class YOLOWorldTracker:
    def __init__(self, weights, vocab, device, conf, imgsz, tracker):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.model.set_classes(list(vocab))
        self.vocab, self.device, self.conf, self.imgsz, self.tracker = list(vocab), device, conf, imgsz, tracker

    def __call__(self, frame_bgr, persist: bool):
        r = self.model.track(frame_bgr, persist=persist, tracker=self.tracker, conf=self.conf,
                             imgsz=self.imgsz, device=self.device, verbose=False)[0]
        b = r.boxes
        if b is None or b.id is None or len(b) == 0:
            return []
        ids = b.id.int().cpu().numpy()
        xyxy, cls, cf = b.xyxy.cpu().numpy(), b.cls.int().cpu().numpy(), b.conf.cpu().numpy()
        names = r.names if isinstance(r.names, dict) else dict(enumerate(r.names))
        return [(int(i), names.get(int(c), str(c)), float(p), *map(float, bb))
                for i, c, p, bb in zip(ids, cls, cf, xyxy)]


# ---- mock backends: CPU-only smoke tests without downloading anything ----------

class MockPatchEncoder:
    """Low-level patch statistics (colour mean + gradient energy), L2-normalised."""

    def __call__(self, frames_bgr):
        out = []
        for f in frames_bgr:
            s = cv2.resize(f, (28 * 8, 16 * 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
            gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
            feats = np.concatenate([s, np.abs(gx)[..., None], np.abs(gy)[..., None], np.ones_like(g)[..., None] * .2], -1)
            p = feats.reshape(16, 8, 28, 8, -1).mean((1, 3))
            p = p - p.mean(-1, keepdims=True) * 0.5
            out.append(p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-6))
        return np.stack(out)


class MockSigLIP:
    logit_scale, logit_bias = 10.0, -5.0

    def __init__(self, dim=64):
        self.P = np.random.default_rng(0).normal(size=(3 * 8 * 8, dim)).astype(np.float32)

    def images(self, frames_bgr):
        e = np.stack([cv2.resize(f, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
                      for f in frames_bgr]) @ self.P
        e -= e.mean(1, keepdims=True)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-6)

    def texts(self, texts):
        e = np.stack([np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
                      .normal(size=self.P.shape[1]) for t in texts]).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


class MockXCLIP:
    num_frames = 8

    def __call__(self, clips_bgr, texts):
        out = []
        for c in clips_bgr:
            motion = np.mean([cv2.absdiff(c[i], c[i - 1]).mean() for i in range(1, len(c))])
            logits = np.zeros(len(texts))
            logits[hash(round(motion, 1)) % (len(texts) - 1)] = motion / 2
            logits[-1] = 1.0
            out.append(np.exp(logits) / np.exp(logits).sum())
        return np.array(out)


class MockTracker:
    """Foreground blobs from a running median background + greedy IoU tracking."""

    def __init__(self):
        self.bg, self.tracks, self.next_id = None, {}, 1

    def __call__(self, frame, persist):
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        if not persist or self.bg is None or self.bg.shape != g.shape:
            self.bg, self.tracks = g.copy(), {}
        fg = (np.abs(g - self.bg) > 25).astype(np.uint8)
        self.bg = 0.98 * self.bg + 0.02 * g
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(fg)
        boxes = [(x, y, x + w, y + h) for x, y, w, h, a in st[1:] if a > 150]
        out, new = [], {}
        for b in boxes:
            best, bid = 0.2, None
            for tid, tb in self.tracks.items():
                iou = _iou(b, tb)
                if iou > best and tid not in new:
                    best, bid = iou, tid
            if bid is None:
                bid, self.next_id = self.next_id, self.next_id + 1
            new[bid] = b
            out.append((bid, "person" if (b[3] - b[1]) > (b[2] - b[0]) else "box", 0.8, *map(float, b)))
        self.tracks = new
        return out


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class Models:
    """Lazy, optional model loading. A failed load disables that method with a warning."""

    def __init__(self, cfg):
        self.cfg, self.rt = cfg, cfg["runtime"]
        self.mock = self.rt["mock_models"]
        self.device = "cpu" if self.mock else pick_device(self.rt["device"])
        self._cache, self.failed = {}, {}

    def _get(self, key, factory):
        if key in self._cache:
            return self._cache[key]
        if key in self.failed:
            return None
        try:
            t0 = time.perf_counter()
            self._cache[key] = factory()
            log(f"loaded {key} on {self.device} ({time.perf_counter() - t0:.1f} s)")
        except Exception as e:  # missing package, no internet, no GPU memory ...
            self.failed[key] = repr(e)
            warnings.warn(f"{key} unavailable, method skipped: {e!r}")
            self._cache[key] = None
        return self._cache[key]

    def patch(self):
        s = self.cfg["deep_gate"]
        return self._get("deep_patch", lambda: MockPatchEncoder() if self.mock else
                         PatchEncoder(s["deep_model"], self.device, self.rt["fp16"], s["deep_input_h"]))

    def siglip(self):
        e = self.cfg["events"]
        return self._get("siglip", lambda: MockSigLIP() if self.mock else
                         SigLIPEncoder(e["siglip_model"], self.device, self.rt["fp16"]))

    def xclip(self):
        e = self.cfg["events"]
        return self._get("xclip", lambda: MockXCLIP() if self.mock else
                         XCLIPScorer(e["xclip_model"], self.device, self.rt["fp16"]))

    def tracker(self):
        e = self.cfg["events"]
        return self._get("tracker", lambda: MockTracker() if self.mock else
                         YOLOWorldTracker(e["detector_model"], e["detector_vocab"], self.device,
                                          e["detector_conf"], e["detector_imgsz"], e["tracker"]))


# =============================================================================
# Deep gate (DINOv2); scan_deep at the bottom runs it over a video
# =============================================================================

class DeepFeatureGate:
    """
    G3: background subtraction in DINOv2 feature space.

    Each patch keeps a background feature vector b and running mean/var of its
    cosine distance d = 1 - <f, b>. A patch is foreground when its distance is
    both an outlier for *that patch* (z > deep_z) and absolutely large
    (d > deep_min_dist). Self-supervised features barely move under global
    lighting changes, sensor noise or compression, but move strongly when an
    object appears, so this filters MOG2's typical false positives while still
    catching low-contrast motion that frame differencing misses.
    Background patches adapt fast; foreground patches adapt very slowly, so an
    object that stops (parked car, dropped bag) is absorbed after a while,
    like MOG2's learning rate.
    """

    def __init__(self, cfg, keep_mask=None):
        self.c, self.keep_mask = cfg, keep_mask
        self.bg = self.mu = self.var = None
        self.n = 0

    def update(self, feats):  # feats: (gh, gw, C) L2-normalised
        c = self.c
        if self.bg is None:
            self.bg = feats.copy()
            self.mu = np.full(feats.shape[:2], 0.02, np.float32)
            self.var = np.full(feats.shape[:2], 0.02 ** 2, np.float32)
            if self.keep_mask is not None:
                self.keep = cv2.resize(self.keep_mask.astype(np.uint8), feats.shape[1::-1],
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                self.keep = np.ones(feats.shape[:2], bool)
        d = 1.0 - np.sum(feats * self.bg, -1)
        z = (d - self.mu) / np.sqrt(self.var + 1e-8)
        warm = self.n < c["deep_warmup"]
        fg = np.zeros_like(d, bool) if warm else (z > c["deep_z"]) & (d > c["deep_min_dist"])
        fg &= self.keep
        a = 0.2 if warm else c["deep_alpha_stats"]
        upd = ~fg
        self.mu[upd] = (1 - a) * self.mu[upd] + a * d[upd]
        self.var[upd] = (1 - a) * self.var[upd] + a * (d[upd] - self.mu[upd]) ** 2
        w = np.where(fg, c["deep_alpha_fg"], 0.5 if warm else c["deep_alpha_bg"])[..., None]
        b = (1 - w) * self.bg + w * feats
        self.bg = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
        self.n += 1
        return float(fg[self.keep].mean()) if self.keep.any() else 0.0, fg


def smooth(x, win):
    return pd.Series(x).rolling(max(1, int(win)), center=True, min_periods=1).mean().to_numpy()


def active_fraction(segs_df, duration):
    return float(segs_df["duration_s"].sum() / duration) if len(segs_df) and duration else 0.0


# =============================================================================
# Stage 3 - event detection
# =============================================================================

def build_windows(segments, info, heartbeat_s, pad_s=0.0):
    """Active windows (segment_id >= 0) + 1-frame idle heartbeat windows (segment_id = -1)."""
    win = [(float(r.start_s), float(r.end_s), int(r.segment_id)) for r in segments.itertuples()]
    if heartbeat_s and heartbeat_s > 0:
        t = 0.0
        while t < info.duration_s:
            if not any(a - pad_s <= t <= b + pad_s for a, b, _ in win):
                win.append((t, t + 1e-3, -1))
            t += heartbeat_s
    return sorted(win)


def iter_window_chunks(video, info, windows, fps, width, chunk):
    """Sequential decode; yields (seg_id, first_chunk, idxs, times, frames) inside the windows."""
    step = max(1, round(info.fps / fps))
    wanted = []
    for a, b, sid in windows:
        f0, f1 = int(a * info.fps), max(int(a * info.fps), int(b * info.fps))
        idxs = list(range(f0, f1 + 1, step)) if sid >= 0 else [f0]
        wanted.extend((i, sid) for i in idxs if i < max(1, info.frame_count))
    wanted.sort()
    cap, cur = cv2.VideoCapture(str(video)), 0
    buf, cur_sid, first = [], None, True

    def emit():
        nonlocal buf, first
        if buf:
            out = (cur_sid, first, [b[0] for b in buf], [b[0] / info.fps for b in buf], [b[1] for b in buf])
            buf, first = [], False
            return out
        return None

    for idx, sid in wanted:
        if sid != cur_sid:
            o = emit()
            if o:
                yield o
            cur_sid, first = sid, True
        while cur < idx:
            if not cap.grab():
                break
            cur += 1
        ok, fr = cap.read()
        if not ok:
            break
        cur += 1
        buf.append((idx, resize_w(fr, min(width, fr.shape[1]))))
        if len(buf) >= chunk:
            yield emit()
    o = emit()
    if o:
        yield o
    cap.release()


def extract_event_features(video, info, segments, cfg, models: Models):
    """Pass 2: decode frames inside segments once; run detector+tracker, SigLIP, X-CLIP."""
    e = cfg["events"]
    windows = build_windows(segments, info, cfg["segments"]["heartbeat_s"])
    use = set(e["methods"])
    trk = models.tracker() if "tracks" in use else None
    sig = models.siglip()  # always needed: it is the index embedding and drives stage 3
    xcl = models.xclip() if "xclip" in use else None
    x_texts = [b["xclip"] for b in EVENT_BANK] + [BACKGROUND_PROMPTS["xclip"]]
    nf = getattr(xcl, "num_frames", e["xclip_frames"]) if xcl else e["xclip_frames"]

    meta, embs, phashes, tracks, xrows = [], [], [], [], []
    tail = []  # carry-over frames so X-CLIP windows can cross chunk boundaries
    t0 = time.perf_counter()
    n_frames = 0
    for sid, first, idxs, ts, frames in iter_window_chunks(video, info, windows, e["event_fps"],
                                                           e["frame_width"], e["chunk"]):
        n_frames += len(frames)
        # E1 detector + tracker (tracker state resets at the start of each segment)
        if trk is not None and sid >= 0:
            for j, (fi, t, fr) in enumerate(zip(idxs, ts, frames)):
                for tid, cls, conf, x1, y1, x2, y2 in trk(fr, persist=not (first and j == 0)):
                    tracks.append({"segment_id": sid, "track_id": f"{sid}:{tid}", "frame_idx": fi,
                                   "timestamp_s": t, "cls": cls, "conf": conf,
                                   "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                   "img_w": fr.shape[1], "img_h": fr.shape[0]})
        # E2 SigLIP embeddings (also the search-index embeddings)
        if sig is not None:
            for k in range(0, len(frames), e["batch_size"]):
                embs.append(sig.images(frames[k:k + e["batch_size"]]))
        for fi, t, fr in zip(idxs, ts, frames):
            meta.append({"frame_idx": fi, "timestamp_s": t, "segment_id": sid})
            phashes.append(np.packbits(phash64(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))).tobytes().hex())
        # E3 X-CLIP clip windows
        if xcl is not None and sid >= 0:
            if first:
                tail = []
            seq = tail + list(zip(idxs, ts, frames))
            clips, centers = [], []
            for s0 in range(0, len(seq) - nf + 1, e["xclip_stride"]):
                w = seq[s0:s0 + nf]
                clips.append([f for _, _, f in w])
                centers.append((w[0][1], w[-1][1]))
            for k in range(0, len(clips), 8):
                probs = xcl(clips[k:k + 8], x_texts)
                for (a, b), p in zip(centers[k:k + 8], probs):
                    xrows.append({"segment_id": sid, "start_s": a, "end_s": b,
                                  **{f"p_{bk['key']}": float(v) for bk, v in zip(EVENT_BANK, p[:-1])},
                                  "p_background": float(p[-1])})
            consumed = (len(seq) - nf) // e["xclip_stride"] * e["xclip_stride"] + e["xclip_stride"] \
                if len(seq) >= nf else 0
            tail = seq[consumed:]
    elapsed = time.perf_counter() - t0
    E = np.concatenate(embs) if embs else np.zeros((len(meta), 0), np.float32)
    meta = pd.DataFrame(meta, columns=["frame_idx", "timestamp_s", "segment_id"])
    meta["phash"] = phashes
    return {"meta": meta, "emb": E, "tracks": pd.DataFrame(tracks), "xclip": pd.DataFrame(xrows),
            "stats": {"frames": n_frames, "seconds": elapsed,
                      "frames_per_s": n_frames / max(elapsed, 1e-9),
                      "skipped_methods": dict(models.failed)}}


def _intervals_from_mask(t, mask, dt):
    out, start = [], None
    for ti, m in zip(t, mask):
        if m and start is None:
            start = ti
        elif not m and start is not None:
            out.append((start, ti))
            start = None
    if start is not None:
        out.append((start, t[-1] + dt))
    return out


def siglip_events(feat, cfg, models):
    e = cfg["events"]
    sig, meta, E = models.siglip(), feat["meta"], feat["emb"]
    if sig is None or E.shape[0] == 0:
        return pd.DataFrame(), None
    texts, owner = [], []
    for i, b in enumerate(EVENT_BANK):
        texts += b["siglip"]
        owner += [i] * len(b["siglip"])
    bg = BACKGROUND_PROMPTS["siglip"]
    T = sig.texts(texts + bg)
    logits = E @ T.T * sig.logit_scale + sig.logit_bias           # (n, prompts)
    owner = np.array(owner)
    ev = np.stack([logits[:, :len(texts)][:, owner == i].max(1) for i in range(len(EVENT_BANK))], 1)
    bgl = logits[:, len(texts):].max(1, keepdims=True)
    margin = ev - bgl                                               # (n, events)
    rows = []
    act = meta["segment_id"].to_numpy() >= 0
    dt = 1.0 / e["event_fps"]
    for sid in np.unique(meta["segment_id"][act]):
        m = np.flatnonzero(meta["segment_id"].to_numpy() == sid)
        t = meta["timestamp_s"].to_numpy()[m]
        for i, b in enumerate(EVENT_BANK):
            sm = smooth(margin[m, i], e["siglip_smooth"])
            for a, z in _intervals_from_mask(t, sm > e["siglip_margin"], dt):
                sel = (t >= a) & (t < z)
                pk = int(np.argmax(np.where(sel, sm, -np.inf)))
                rows.append({"source": "siglip", "key": b["key"], "segment_id": int(sid), "start_s": a,
                             "end_s": z, "score": float(1 / (1 + np.exp(-(sm[pk] - e["siglip_margin"])))),
                             "peak_s": float(t[pk]), "detail": b["siglip"][0]})
    # per-frame top tag (stored with the index; free open-vocabulary hint)
    top = margin.argmax(1)
    tags = [EVENT_BANK[k]["key"] if margin[j, k] > e["siglip_margin"] else "" for j, k in enumerate(top)]
    return pd.DataFrame(rows), {"margin": margin, "tags": tags}


def xclip_events(feat, cfg):
    e, X = cfg["events"], feat["xclip"]
    if X is None or X.empty:
        return pd.DataFrame()
    rows = []
    for sid, g in X.groupby("segment_id"):
        g = g.sort_values("start_s")
        for b in EVENT_BANK:
            p = g[f"p_{b['key']}"].to_numpy()
            hit = (p > e["xclip_threshold"]) & (p > g["p_background"].to_numpy())
            cur = None
            for (a, z, pv, h) in zip(g["start_s"], g["end_s"], p, hit):
                if h and cur and a <= cur["end_s"] + e["merge_gap_s"]:
                    cur["end_s"] = max(cur["end_s"], z)
                    if pv > cur["score"]:
                        cur["score"], cur["peak_s"] = float(pv), (a + z) / 2
                elif h:
                    if cur:
                        rows.append(cur)
                    cur = {"source": "xclip", "key": b["key"], "segment_id": int(sid), "start_s": a,
                           "end_s": z, "score": float(pv), "peak_s": (a + z) / 2, "detail": b["xclip"]}
            if cur:
                rows.append(cur)
    return pd.DataFrame(rows)


def track_events(tracks, cfg):
    """E1 rules on top of open-vocabulary tracks. Distances are in box heights (scale-free)."""
    e = cfg["events"]
    if tracks is None or tracks.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows, summ = [], []
    dt = 1.0 / e["event_fps"]
    tr = tracks.assign(cx=(tracks.x1 + tracks.x2) / 2, cy=(tracks.y1 + tracks.y2) / 2,
                       h=(tracks.y2 - tracks.y1).clip(lower=1), w=(tracks.x2 - tracks.x1).clip(lower=1))
    for tid, g in tr.groupby("track_id"):
        g = g.sort_values("timestamp_s")
        cls = g["cls"].mode().iat[0]
        t = g["timestamp_s"].to_numpy()
        dur = t[-1] - t[0] + dt
        size = float(np.median(np.maximum(g.h, g.w)))
        ctr = g[["cx", "cy"]].to_numpy()
        radius = float(np.sqrt(((ctr - ctr.mean(0)) ** 2).sum(1).mean())) / size
        summ.append({"track_id": tid, "segment_id": int(g.segment_id.iat[0]), "cls": cls, "start_s": t[0],
                     "end_s": t[-1] + dt, "duration_s": dur, "mean_conf": float(g.conf.mean()),
                     "radius_rel": radius})
        if dur < e["min_track_s"]:
            continue
        base = {"source": "tracks", "segment_id": int(g.segment_id.iat[0]), "detail": f"{cls} {tid}"}
        key = "vehicle_appears" if cls in e["vehicles"] else f"{cls.replace(' ', '_')}_appears"
        rows.append({**base, "key": key, "start_s": t[0], "end_s": t[0] + 1.0,
                     "score": float(g.conf.mean()), "peak_s": t[0]})
        if cls == "person":
            if dur >= e["loiter_s"] and radius < e["loiter_radius"]:
                rows.append({**base, "key": "person_loitering", "start_s": t[0], "end_s": t[-1] + dt,
                             "score": min(1.0, dur / (2 * e["loiter_s"])), "peak_s": (t[0] + t[-1]) / 2})
            if len(t) >= 3:
                sp = np.r_[0, np.linalg.norm(np.diff(ctr, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)] / size
                sp = smooth(sp, max(1, round(1.0 / dt)))
                for a, z in _intervals_from_mask(t, sp > e["run_speed"], dt):
                    if z - a >= 0.75:
                        k = int(np.argmax(np.where((t >= a) & (t < z), sp, -1)))
                        rows.append({**base, "key": "running", "start_s": a, "end_s": z,
                                     "score": float(min(1.0, sp[k] / (2 * e["run_speed"]))), "peak_s": t[k]})
        elif cls in e["carried_objects"] and dur >= e["static_s"] and radius < e["static_radius"]:
            rows.append({**base, "key": "object_left", "start_s": t[0], "end_s": t[-1] + dt,
                         "score": float(g.conf.mean()), "peak_s": t[0] + e["static_s"]})
    # crowd: many persons at once
    per = tr[tr.cls == "person"].groupby(["segment_id", "timestamp_s"]).track_id.nunique().reset_index()
    for sid, g in per.groupby("segment_id"):
        t, n = g.timestamp_s.to_numpy(), g.track_id.to_numpy()
        for a, z in _intervals_from_mask(t, n >= e["crowd_n"], dt):
            if z - a >= e["crowd_min_s"]:
                sel = (t >= a) & (t < z)
                rows.append({"source": "tracks", "segment_id": int(sid), "detail": f"max {n[sel].max()} people",
                             "key": "crowd", "start_s": a, "end_s": z,
                             "score": float(min(1.0, n[sel].max() / (2 * e["crowd_n"]))),
                             "peak_s": float(t[sel][np.argmax(n[sel])])})
    return pd.DataFrame(rows), pd.DataFrame(summ)


def merge_events(ev_list, gap):
    ev = pd.concat([x for x in ev_list if x is not None and not x.empty], ignore_index=True) \
        if any(x is not None and not x.empty for x in ev_list) else pd.DataFrame()
    if ev.empty:
        return pd.DataFrame(columns=["event_id", "source", "key", "segment_id", "start_s", "end_s",
                                     "score", "peak_s", "detail", "support", "supported_by"])
    out = []
    for (src, key, sid), g in ev.sort_values("start_s").groupby(["source", "key", "segment_id"]):
        cur = None
        for r in g.to_dict("records"):
            if cur and r["start_s"] <= cur["end_s"] + gap and not key.endswith("_appears"):
                cur["end_s"] = max(cur["end_s"], r["end_s"])
                if r["score"] > cur["score"]:
                    cur["score"], cur["peak_s"] = r["score"], r["peak_s"]
            else:
                if cur:
                    out.append(cur)
                cur = dict(r)
        out.append(cur)
    ev = pd.DataFrame(out).sort_values(["start_s", "key"]).reset_index(drop=True)
    # cross-method support: how many distinct sources saw the same key overlapping in time
    sup, by = [], []
    for r in ev.itertuples():
        o = ev[(ev.key == r.key) & (ev.start_s < r.end_s + gap) & (ev.end_s > r.start_s - gap)]
        srcs = sorted(o.source.unique())
        sup.append(len(srcs))
        by.append("+".join(srcs))
    ev["support"], ev["supported_by"] = sup, by
    ev.insert(0, "event_id", range(len(ev)))
    return ev


def importance_curve(meta, scores, fused_col, events, track_summary, cfg):
    """Per candidate frame importance in [0, 1]: motion, events and track births."""
    t = meta["timestamp_s"].to_numpy()
    mot = np.interp(t, scores["timestamp_s"], scores[fused_col]) if fused_col in scores else np.zeros_like(t)
    imp = np.clip(mot / 3.0, 0, 1) * 0.5
    for r in events.itertuples():
        m = (t >= r.start_s) & (t <= r.end_s)
        w = r.score * (0.6 + 0.2 * min(r.support, 3) - 0.2) if r.source != "tracks" or \
            not r.key.endswith("_appears") else r.score * 0.8
        imp[m] = np.maximum(imp[m], np.clip(w, 0, 1))
    if track_summary is not None and not track_summary.empty:
        for a in track_summary.loc[track_summary.duration_s >= cfg["events"]["min_track_s"], "start_s"]:
            imp[np.abs(t - a) <= 0.5] = 1.0
    imp[meta["segment_id"].to_numpy() < 0] = 0.0
    return imp


def event_anchors(meta, events, track_summary, cfg):
    """Candidate-row indices that must be represented: event peaks and track births."""
    t, sid = meta["timestamp_s"].to_numpy(), meta["segment_id"].to_numpy()
    anchors = set()

    def nearest(x, s):
        m = np.flatnonzero(sid == s)
        if m.size:
            anchors.add(int(m[np.argmin(np.abs(t[m] - x))]))

    for r in events.itertuples():
        if r.support >= 2 or r.source == "tracks" or r.score >= 0.7:
            nearest(r.peak_s, r.segment_id)
    if track_summary is not None and not track_summary.empty:
        for r in track_summary[track_summary.duration_s >= cfg["events"]["min_track_s"]].itertuples():
            nearest(min(r.start_s + 0.5, r.end_s), r.segment_id)
    return anchors


# =============================================================================
# Stage 4 - adaptive sampling
# =============================================================================

def _budget(dur_s, mean_imp, c):
    b = round(c["rate_per_min"] * dur_s / 60.0 * (0.5 + mean_imp))
    return int(np.clip(b, c["min_per_segment"], c["max_per_segment"]))


def sample_phash(meta, c, **_):
    """S0 baseline: keep a frame when its pHash differs enough from the last kept one."""
    keep = []
    for sid, g in meta.groupby("segment_id", sort=False):
        idx = g.index.to_numpy()
        if sid < 0:
            keep += list(idx)
            continue
        hs = [np.unpackbits(np.frombuffer(bytes.fromhex(h), np.uint8)) for h in g.phash]
        last, last_t = None, -1e9
        for j, (i, t) in enumerate(zip(idx, g.timestamp_s)):
            if last is None or (hamming(hs[j], hs[last]) >= c["phash_change"] and t - last_t >= c["min_gap_s"]) \
                    or t - last_t >= c["max_gap_s"]:
                keep.append(i)
                last, last_t = j, t
    return sorted(keep)


def sample_novelty(meta, c, E, **_):
    """S1: online semantic novelty. Keep a frame if it is far from EVERY frame kept so far in the
    segment (not just the last one, which avoids A-B-A repeats). Threshold adapts per segment."""
    keep = []
    for sid, g in meta.groupby("segment_id", sort=False):
        idx = g.index.to_numpy()
        if sid < 0:
            keep += list(idx)
            continue
        X = E[idx]
        step = 1 - np.sum(X[1:] * X[:-1], 1) if len(X) > 1 else np.zeros(1)
        tau = max(c["novelty_tau"], float(np.quantile(step, c["novelty_adaptive_q"]))) if len(step) else c["novelty_tau"]
        kept, last_t = [], -1e9
        for j, (i, t) in enumerate(zip(idx, g.timestamp_s)):
            nov = 1.0 if not kept else 1 - float(np.max(X[kept] @ X[j]))
            if (nov > tau and t - last_t >= c["min_gap_s"]) or t - last_t >= c["max_gap_s"]:
                kept.append(j)
                last_t = t
        keep += list(idx[kept])
    return sorted(keep)


def sample_kcenter(meta, c, E, imp, anchors, **_):
    """S2: importance-weighted k-center greedy in (semantic + temporal) space.
    Seeds with event anchors, then repeatedly adds the frame worst covered by the selection,
    weighted by importance. Stops at the budget or once every frame is within `kcenter_radius`
    (adaptive budget: a static scene needs 1 frame, a busy one gets more)."""
    keep = []
    for sid, g in meta.groupby("segment_id", sort=False):
        idx = g.index.to_numpy()
        if sid < 0:
            keep += list(idx)
            continue
        X, t, w = E[idx], g.timestamp_s.to_numpy(), imp[idx]
        dur = max(t[-1] - t[0], 1e-6)
        B = _budget(dur, float(w.mean()), c)
        lam = c["kcenter_lambda_t"]

        def dist_to(j):
            return (1 - lam) * (1 - X @ X[j]) + lam * np.abs(t - t[j]) / dur

        sel = [int(np.flatnonzero(idx == a)[0]) for a in anchors if a in set(idx)]
        sel = sel or [int(np.argmax(w))]
        mind = np.min(np.stack([dist_to(j) for j in sel]), 0)
        while len(sel) < max(B, len(sel)):
            if mind.max() < c["kcenter_radius"]:
                break
            j = int(np.argmax(mind * (0.5 + 0.5 * w)))
            sel.append(j)
            mind = np.minimum(mind, dist_to(j))
        keep += list(idx[sorted(set(sel))])
    return sorted(keep)


def sample_importance(meta, c, E, imp, anchors, **_):
    """S3: stratified inverse-CDF sampling of the importance curve (dense where things happen),
    plus anchors, then near-duplicate removal in embedding space."""
    keep = []
    for sid, g in meta.groupby("segment_id", sort=False):
        idx = g.index.to_numpy()
        if sid < 0:
            keep += list(idx)
            continue
        w, t = imp[idx] + 0.05, g.timestamp_s.to_numpy()
        B = _budget(t[-1] - t[0], float(imp[idx].mean()), c)
        cdf = np.cumsum(w) / w.sum()
        cand = [int(np.flatnonzero(idx == a)[0]) for a in anchors if a in set(idx)]
        cand += [int(np.searchsorted(cdf, (k + 0.5) / B)) for k in range(B)]
        sel = []
        for j in dict.fromkeys(min(j, len(idx) - 1) for j in cand):
            if not sel or E.shape[1] == 0 or float(np.max(E[idx[sel]] @ E[idx[j]])) < c["dedup_sim"]:
                sel.append(j)
        # never return fewer than min_per_segment: top up with the most important unused frames,
        # spread out by keeping them at least a quarter of the segment apart where possible
        need = min(c["min_per_segment"], len(idx)) - len(sel)
        for j in np.argsort(-imp[idx]):
            if need <= 0:
                break
            if j not in sel and all(abs(t[j] - t[s]) >= (t[-1] - t[0]) / 4 for s in sel):
                sel.append(int(j))
                need -= 1
        keep += list(idx[sorted(sel)])
    return sorted(keep)


SAMPLERS = {"phash": sample_phash, "novelty": sample_novelty, "kcenter": sample_kcenter,
            "importance": sample_importance}


def evaluate_sampling(meta, E, keep, events, track_summary, info, c):
    act = meta["segment_id"].to_numpy() >= 0
    keep = np.array(sorted(keep), int)
    k_act = keep[act[keep]] if len(keep) else keep
    t, sid = meta["timestamp_s"].to_numpy(), meta["segment_id"].to_numpy()
    cov, red = [], []
    for s in np.unique(sid[act]):
        cand = np.flatnonzero(sid == s)
        sel = k_act[sid[k_act] == s]
        if E.shape[1] and len(sel):
            S = E[cand] @ E[sel].T
            cov.append(S.max(1))
            if len(sel) > 1:
                SS = E[sel] @ E[sel].T
                np.fill_diagonal(SS, -1)
                red.append(SS.max(1))
    cov = np.concatenate(cov) if cov else np.array([np.nan])
    red = np.concatenate(red) if red else np.array([np.nan])
    nanmean = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
    kt = t[k_act]
    ev_hit = [bool(np.any((kt >= r.start_s) & (kt <= r.end_s))) for r in events.itertuples()]
    tr = track_summary[track_summary.duration_s >= 1.0] if track_summary is not None and not track_summary.empty \
        else pd.DataFrame()
    tr_hit = [bool(np.any((kt >= r.start_s) & (kt <= r.end_s))) for r in tr.itertuples()]
    naive = int(info.duration_s * c["naive_fps"])
    return {"keyframes": int(len(keep)), "active_keyframes": int(len(k_act)),
            "keyframes_per_min": len(keep) / max(info.duration_s / 60, 1e-9),
            "reduction_vs_naive_x": naive / max(1, len(keep)),
            "coverage_mean": nanmean(cov),
            "coverage_p10": float(np.nanquantile(cov, 0.1)) if np.isfinite(cov).any() else float("nan"),
            "redundancy_mean": nanmean(red),
            "event_recall": float(np.mean(ev_hit)) if ev_hit else float("nan"),
            "track_recall": float(np.mean(tr_hit)) if tr_hit else float("nan")}


# =============================================================================
# Outputs
# =============================================================================

def export_keyframes(video, info, meta, keep, out_dir, name, camera_id):
    d = out_dir / "keyframes" / name
    d.mkdir(parents=True, exist_ok=True)
    rows = meta.loc[keep]
    frames = dict(read_frames(video, rows.frame_idx))
    files = []
    for r in rows.itertuples():
        fn = f"{camera_id}_{format_hms(r.timestamp_s).replace(':', '-')}_" + \
             ("idle" if r.segment_id < 0 else f"seg{r.segment_id:03d}") + ".jpg"
        if r.frame_idx in frames:
            cv2.imwrite(str(d / fn), frames[r.frame_idx], [cv2.IMWRITE_JPEG_QUALITY, 90])
        files.append(fn)
    return files


def write_index_records(out, meta, keep, E, events, track_sum, camera_id, video_name, t_start=None):
    """Hand-off to captioning / indexing: out/index_records.jsonl (one record per keyframe) and
    out/index_embeddings.npy (row `embedding_row` of a record = its SigLIP embedding)."""
    out = Path(out)
    np.save(out / "index_embeddings.npy", E[keep].astype(np.float16) if E.shape[1] else E[:0])
    with open(out / "index_records.jsonl", "w") as f:
        for row_i, (ci, r) in enumerate(meta.loc[keep].iterrows()):
            act = events[(events.start_s <= r.timestamp_s) & (events.end_s >= r.timestamp_s)] if len(events) else events
            objs = track_sum[(track_sum.start_s <= r.timestamp_s) & (track_sum.end_s >= r.timestamp_s)] \
                if len(track_sum) else track_sum
            rec = {"camera_id": camera_id, "video": video_name, "frame_idx": int(r.frame_idx),
                   "timestamp_s": round(float(r.timestamp_s), 3), "segment_id": int(r.segment_id),
                   "wallclock": (t_start + timedelta(seconds=float(r.timestamp_s))).isoformat() if t_start else None,
                   "importance": round(float(r.get("importance", 0.0)), 3), "embedding_row": row_i,
                   "events": sorted(set(act.key)) if len(act) else [],
                   "objects": objs.cls.value_counts().to_dict() if len(objs) else {},
                   "siglip_tag": r.get("siglip_tag", "")}
            f.write(json.dumps(rec) + "\n")
    return out / "index_records.jsonl"


def plot_timeline(path, scores, gate_results, primary, events, keeps, info):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    gates = [m for m in ("diff", "mog2", "deep", "fused") if m in gate_results]
    n = len(gates) + 2
    fig, axes = plt.subplots(n, 1, figsize=(15, 1.9 * n), sharex=True)
    t = scores.timestamp_s / 60
    for ax, m in zip(axes, gates):
        r = gate_results[m]
        y = scores[r["col"]] * (1 if m == "fused" else 100)
        ax.plot(t, y, lw=.7)
        ax.axhline(r["start"] * (1 if m == "fused" else 100), color="r", ls="--", lw=.8)
        for s in r["segments"].itertuples():
            ax.axvspan(s.start_s / 60, s.end_s / 60, color="C1", alpha=.18)
        ax.set_ylabel("ratio" if m == "fused" else "% px")
        ax.set_title(f"{m}: {len(r['segments'])} segments, "
                     f"{active_fraction(r['segments'], info.duration_s):.1%} active"
                     + ("  [primary]" if m == primary else ""), fontsize=9, loc="left")
        ax.grid(alpha=.25)
    ax = axes[-2]
    keys = sorted(events.key.unique()) if len(events) else []
    colors = {"tracks": "C0", "siglip": "C2", "xclip": "C3"}
    for r in events.itertuples():
        y = keys.index(r.key)
        ax.plot([r.start_s / 60, r.end_s / 60], [y, y], lw=5 if r.support > 1 else 2.5,
                color=colors.get(r.source, "k"), solid_capstyle="butt", alpha=.8)
    ax.set_yticks(range(len(keys)), keys, fontsize=7)
    ax.set_title("events (blue tracks, green SigLIP, red X-CLIP; thick = supported by 2+ methods)",
                 fontsize=9, loc="left")
    ax.grid(alpha=.25)
    ax = axes[-1]
    for i, (name, kt) in enumerate(keeps.items()):
        ax.vlines(np.asarray(kt) / 60, i - .35, i + .35, lw=.8)
    ax.set_yticks(range(len(keeps)), list(keeps), fontsize=8)
    ax.set_title("keyframes per sampler", fontsize=9, loc="left")
    ax.set_xlabel("video time (min)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =============================================================================
# Deep gate over a whole video
# =============================================================================


def scan_deep(video_path, scores: pd.DataFrame, cfg: dict, models: "Models",
              cache_dir: str | Path | None = None) -> tuple[np.ndarray | None, dict]:
    """Deep-gate score for every row of `scores` (same samples as stage 1), added to them as score_deep.

    How it works (DeepFeatureGate): every frame is cut into 14x14-pixel patches and DINOv2
    turns each patch into a feature vector. Each patch keeps a slowly updated "background" vector; a patch
    counts as moving when it differs from its background much more than usual. Score = share of patches
    moving. Runs at `deep_fps` (2/s) and is held constant in between.
    
    Needs torch + transformers and, realistically, a GPU. If the model can't load, scan_deep returns None
    and the notebook simply leaves the deep gate out.
    Known weakness (seen on cafe_demo.mp4): the first frames become the background, so people present at
    the start count as "change" until the slow update absorbs them (about a minute).


    cfg:    DEFAULT_CONFIG plus overrides; uses cfg["deep_gate"] and cfg["runtime"]["mock_models"]
    models: Models(cfg); models.patch() loads DINOv2 (or a mock)
    Returns (score array or None if the model is unavailable, stats dict). Cached in cache_dir.
    """
    s = cfg["deep_gate"]
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
            small = resize_w(frame, 320)
            gate = DeepFeatureGate(s, polygon_mask(small.shape[:2], s["ignore_polygons"]))
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
