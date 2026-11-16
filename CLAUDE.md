# CLAUDE.md

Part of the REWIND team project: make long fixed-camera (CCTV) video searchable by sending only a few
well-chosen frames to expensive models (captioning, indexing) instead of every frame.

`README.md` explains every script, parameter and output in detail. Read it before changing the pipeline.

## Environment

- Conda env **`motion-gating`** (defined in `environment.yml`, Python 3.11). Always run things inside it:
  `conda activate motion-gating`. Create it once with `conda env create -f environment.yml`.
- Jupyter kernel is also called `motion-gating` (`python -m ipykernel install --user --name motion-gating`).
- Windows machine. The env pins OpenBLAS instead of MKL on purpose: with MKL, Jupyter kernels crashed on
  Windows. Don't remove that pin.
- The env uses `opencv-python-headless`. `rewind_pipeline_Muzz.py` asks for `opencv-python`, and
  `ultralytics` also installs it. Having both in one env can break `cv2`, so check `import cv2` after adding
  the deep-learning packages, or put them in a separate env.

## Commands

```bash
python meva_30min.py        # download the test video -> video/meva_30min.mp4 (6 x 5-min MEVA clips joined)
python motion_gating.py     # stage 1: motion scores + segments (default video: video/meva_30min.mp4)
python keyframes.py         # stage 2: keyframes per segment
# or open explore.ipynb (kernel motion-gating) and Run All: runs both stages and shows the results
```

There are no tests. Check a change by re-running the notebook and comparing the numbers. The baseline on
`meva_30min.mp4` is 109 keyframes instead of 1,801 at 1 fps (16.5x fewer).

## Layout

| Path | Role |
|---|---|
| `motion_gating.py` | Stage 1: frame diff + MOG2 (with an NCC lighting filter), adaptive threshold, hysteresis segments |
| `keyframes.py` | Stage 2: first / peak / change (4x4 tile pHash) / maxgap keyframes, dedup, idle heartbeat |
| `video_utils.py`, `visualize.py` | Frame reading, and plots/GIFs for the notebook |
| `explore.ipynb` | Imports the modules above, runs both stages, shows the results. Committed **with** outputs (~3 MB) |
| `rewind_pipeline_Muzz.py` | Team member's standalone v2 pipeline (see below). Not committed yet |
| `video/`, `outputs/` | Git-ignored. Videos and results stay local |

The slow decode pass is cached in `outputs/<video>/cache/scan_<hash>.npz`. Changing a scan parameter
invalidates it; changing a segment or keyframe parameter does not.

## `rewind_pipeline_Muzz.py` (v2, standalone)

A single self-contained script, written with an LLM from this repo's v1 logic. It does **not** import
`motion_gating.py`, `keyframes.py` or the notebook. It re-implements diff + MOG2 as baselines and adds deep
models:

1. **Gating:** diff, MOG2, and DINOv2 patch-feature background change, fused by a 2-of-3 vote.
2. **Event detection** (only inside segments): YOLO-World + ByteTrack track rules (appears, loitering,
   running, object_left, crowd), SigLIP zero-shot frame tags, X-CLIP zero-shot 8-frame clip tags.
3. **Adaptive sampling:** pHash (v1-style baseline), semantic novelty, importance-weighted k-center (primary),
   importance inverse-CDF. All four are scored on coverage, redundancy, event recall and track recall.

Hand-off to the next stage: `index_records.jsonl` + `index_embeddings.npy` (SigLIP), one record per keyframe.

```bash
pip install torch transformers ultralytics lap      # extra deps, not in environment.yml
python rewind_pipeline_Muzz.py video/meva_30min.mp4 --mock-models --out outputs_v2   # CPU plumbing check, numbers meaningless
python rewind_pipeline_Muzz.py video/meva_30min.mp4 --camera-id meva01 --start-time 2026-11-15T14:00:00 --out outputs_v2 [--gt gt.csv]
```

- Use `--out outputs_v2` (or any folder other than `outputs`). It writes to `<out>/<video stem>/` and would
  otherwise overwrite v1's `motion_scores.csv` and write into v1's `keyframes/` folder.
- Built for a GPU (an L4). The DINOv2 gate runs on the whole video, so it is slow on CPU. `--no-deep-gate`
  skips it.
- `--gt` expects a CSV with `start_s,end_s[,label]`. Without it there is no ground-truth measurement, only
  self-consistency metrics.
- If a model fails to load, its method is skipped with a warning and the run continues. Check the log.
