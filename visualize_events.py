"""Drawing helpers for the GPU stages of pipeline.ipynb (tracker boxes, headers, GIFs).
From the teammate's notebook (rewind_pipeline_pictures.ipynb). Display only: nothing here changes a result."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import cv2
import numpy as np

from visualize import thumb


def track_color(tid) -> tuple[int, int, int]:
    """A stable colour per track id."""
    h = int(hashlib.md5(str(tid).encode()).hexdigest()[:6], 16)
    return (60 + h % 196, 60 + (h >> 8) % 196, 60 + (h >> 16) % 196)


def boxes_for(tracks, frame_idx):
    """Tracker rows of one frame (empty if there are no tracks)."""
    return tracks[tracks.frame_idx == frame_idx] if len(tracks) else tracks


def draw_boxes(img, rows):
    """rows: raw tracker rows of one frame. Coordinates are in the img_w-wide event frame, rescaled to img."""
    out = img.copy()
    for r in rows.itertuples():
        s = img.shape[1] / r.img_w
        c = track_color(r.track_id)
        p1, p2 = (int(r.x1 * s), int(r.y1 * s)), (int(r.x2 * s), int(r.y2 * s))
        cv2.rectangle(out, p1, p2, c, 2)
        cv2.putText(out, f"{r.cls} {str(r.track_id).split(':')[-1]}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return out


def header(text, width):
    """White strip with black text, to put a title above a picture."""
    band = np.full((22, width, 3), 255, np.uint8)
    cv2.putText(band, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return band


def name_tile(name, sub, height, width=110):
    """Small white tile with a name and a grey sub-line (row labels in sampler comparisons)."""
    t = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(t, name, (4, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(t, sub, (4, height // 2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
    return t


def embed_gif(frames_bgr, path, fps_play, caption, width=256) -> str:
    """Save frames as a GIF and return an HTML <figure> that embeds it."""
    from PIL import Image as PILImage
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ims = [PILImage.fromarray(cv2.cvtColor(thumb(f, width), cv2.COLOR_BGR2RGB)) for f in frames_bgr]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps_play), loop=0)
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure style="display:inline-block;margin:4px"><img src="data:image/gif;base64,{b64}" '
            f'width="{width}"><figcaption style="font-size:11px">{caption}</figcaption></figure>')
