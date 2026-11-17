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
- Also runs on the team's Linux VM (2x L4). There, envs live on the 5T disk (`/mnt/5T_storage/envs/...`, created with
  `conda env create -p`), and pip / Hugging Face caches must not go to the nearly full root disk.
- GPU stages: `pip install -r requirements-gpu.txt`. `ultralytics` installs `opencv-python` next to
  `opencv-python-headless`; having both can break `cv2`, so check `import cv2` afterwards.

## Commands

```bash
python meva_30min.py        # download the test video -> video/meva_30min.mp4 (6 x 5-min MEVA clips joined)
python motion_gating.py     # stage 1: motion scores + segments (default video: video/meva_30min.mp4)
python keyframes.py         # stage 2: keyframes per segment
# or open pipeline.ipynb (kernel motion-gating) and Run All: every stage (GPU stages skip cleanly without models)
```

There are no tests. Check a change by re-running the notebook and comparing the numbers. The old baseline on
`meva_30min.mp4` was 109 keyframes instead of 1,801 at 1 fps (16.5x fewer), measured *before* `q_noise=0.5` and
`mog2_color=True` became defaults; re-measure it. For a CPU plumbing check of the whole notebook, set
`"mock_models": True` in its config cell.

## Layout

| Path | Role |
|---|---|
| `motion_gating.py` | Stage 1: frame diff + MOG2 (colour, NCC lighting filter), adaptive threshold on the quieter half, hysteresis segments, fused gate (`add_fused`) |
| `deep_gate.py` | Optional DINOv2 gate: adds `score_deep` to stage 1's scores (uses `rewind_pipeline.DeepFeatureGate`) |
| `keyframes.py` | Stage 2: first / peak / change (4x4 tile pHash) / maxgap keyframes, dedup, idle heartbeat |
| `evaluate.py` | Ground truth (`start_s,end_s[,label]`): gate precision/recall/event recall, keyframe hits, gate agreement |
| `rewind_pipeline.py` | v2 (Muzayyan, LLM-written): models, events, samplers, `write_index_records`. Imported by the notebook; also a standalone CLI |
| `video_utils.py`, `visualize.py`, `visualize_events.py` | Frame reading, plots/GIFs, tracker-box drawing |
| `pipeline.ipynb` | Main notebook (merged v1 + v2). Committed **without** outputs |
| `explore.ipynb` | Older v1-only notebook, committed with MEVA outputs (~3 MB) |
| `video/`, `outputs/` | Git-ignored. Videos and results stay local |

The slow decode pass is cached in `outputs/<video>/cache/scan_<hash>.npz`. Changing a scan parameter
invalidates it; changing a segment or keyframe parameter does not.

## `rewind_pipeline.py` (v2) and the merge

Written with an LLM from v1's logic; it re-implemented diff + MOG2 inside one 1,600-line script. In the merge,
`pipeline.ipynb` uses v1's modules for stage 1 and the CPU keyframes, and imports `rewind_pipeline.py` only for
the DINOv2 gate, the models, stage 3 (YOLO-World + ByteTrack track rules, SigLIP zero-shot tags, X-CLIP) and
stage 4 (samplers phash / novelty / kcenter / importance, scored together with v1's keyframes).
Taken over from v2 into v1: the quieter-half threshold (`q_noise`), colour MOG2, the fused vote, GT evaluation.

- Hand-off to the next stage: `outputs/<video>/events/index_records.jsonl` + `index_embeddings.npy` (SigLIP,
  768-d for `siglip-base-patch16-224`), one record per keyframe of the primary sampler.
- If a model fails to load, its method is skipped with a warning (`models.failed`). SigLIP needs `sentencepiece`.
- Known weak spots (café video): track IDs switch in crowds, `crowd`/`loitering` fire constantly in busy scenes,
  X-CLIP's softmax over 13 prompts rarely passes `xclip_threshold` 0.35, the deep gate's first frames become its
  background (people there at the start count as change for ~1 min).
- Standalone CLI still works: `python rewind_pipeline.py video/x.mp4 --out outputs_v2 [--mock-models] [--gt gt.csv]`
  (use an `--out` other than `outputs`, or it overwrites v1's files).
