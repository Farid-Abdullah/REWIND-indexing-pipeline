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
- GPU stages: `pip install torch transformers accelerate ultralytics lap sentencepiece protobuf` (captions with
  Qwen3-VL need `transformers>=4.57`). `ultralytics` installs
  `opencv-python` next to `opencv-python-headless`; having both can break `cv2`, so check `import cv2` afterwards.

## Commands

```bash
# main way: pipeline.ipynb (config cell, Run All: pre-flight check, fresh start, phases cpu -> gpu -> vlm_clips ->
# vlm_frames), then search.ipynb (index, test queries from the GT, scores -> outputs/batch/search/), then
# explore.ipynb (Run All) to look at the results. The user prefers notebooks over terminal commands.
python datasets.py          # list MEVA / UCA videos, hours and annotations under DATA_ROOT
python batch.py --fresh     # same as pipeline.ipynb from a terminal -> outputs/batch/
python captions.py --mock      # stage 5 on every video folder with a fake captioner (plumbing check)
python motion_gating.py v.mp4 && python keyframes.py v.mp4   # stages 1-2 on one video
# notebooks: pipeline.ipynb runs the batch (no pictures), explore.ipynb shows the results; kernel "rewind (5T)" on the VM
```

There are no tests. Check a change by re-running the batch and comparing `outputs/batch/batch_summary.csv`.
Notebook plumbing check off the VM: `REWIND_DATA_ROOT=… REWIND_OUT_ROOT=… REWIND_SELECTION=… REWIND_MOCK=1 jupyter
nbconvert --to notebook --execute pipeline.ipynb --output-dir /tmp`, then the same for `explore.ipynb`.
For a plumbing check without the VM's data, `REWIND_DATA_ROOT`, `REWIND_OUT_ROOT`, `REWIND_SELECTION` and
`REWIND_MOCK=1` point the notebooks at a fake dataset tree and fake GPU models (search: fake SigLIP/CLIP, TF-IDF text). The old baseline on
`meva_30min.mp4` was 109 keyframes instead of 1,801 at 1 fps (16.5x fewer), measured *before* `q_noise=0.5` and
`mog2_color=True` became defaults; re-measure it. For a CPU plumbing check of the whole notebook, set
`"mock_models": True` in its config cell.

## Layout

| Path | Role |
|---|---|
| `pipeline.ipynb` | The run: pre-flight check (GPU via torch, `sentencepiece`, `ultralytics`, VLM folder), `FRESH_RUN` (`batch.clean_outputs`), then one cell per phase via `batch.run_batch`; logs to `outputs/batch/run_log.txt`; prints only. Committed **without** outputs |
| `explore.ipynb` | Reads `outputs/batch/` and shows everything (overview, per video, MEVA multi-camera, GT misses, events, captions next to their frames and labels, caption speed / hand check, run times, search scores + example results). Committed **without** outputs |
| `batch.py` | Phases `cpu`, `gpu`, `vlm_clips`, `vlm_frames` (`"vlm"` = both), each over all videos; cpu/gpu skip finished videos, caption phases resume per item; a list in `CFG["vlm"]["device"]` = one spawned worker per GPU (`caption_parallel`, videos split by duration, log lines relayed through a queue); `clean_outputs`, `make_logger`, per-video `error_<phase>.json`, `batch_summary.csv` |
| `datasets.py` | MEVA / UCA discovery, ~1 h selection per dataset, ground truth (MEVA `*.activities.yml` frames / fps; UCA sentences) |
| `motion_gating.py` | Stage 1: frame diff + MOG2 (colour, NCC lighting filter), adaptive threshold on the quieter half, hysteresis segments, fused gate (`add_fused`); ground-truth checks (`gate_table`, `keyframe_hits`, `gate_agreement`) |
| `keyframes.py` | Stage 2: first / peak / change (4x4 tile pHash) / maxgap keyframes, dedup, idle heartbeat |
| `gpu_stages.py` | From Muzayyan's v2 (LLM-written): `Models` (lazy, skip on failure), DINOv2 gate (`scan_deep`), stage 3 events, stage 4 samplers, `write_index_records` |
| `captions.py` | Stage 5: Qwen-VL JSON captions per selected frame and per clip (segment cut into <= 60 s), resume by `settings_hash`, `MockVL`, `caption_review.csv` hand-check sheet |
| `caption_eval.py` | Caption quality on saved outputs (explore.ipynb section 6): `detector_crosscheck` (caption mentions vs YOLO tracks per clip), `label_similarity` (GT label vs covering clip caption, `TextEmbedder` = bge-small with TF-IDF fallback, shared with the search), `make_review_sheets` / `review_results` (hand check in `outputs/batch/clip_review/`, never overwrites filled sheets; clip ids written as `c001.0` so Excel keeps them) |
| `search.py` / `search.ipynb` | Search + its evaluation: `SearchIndex` (clip captions via `ce.TextEmbedder`, stage-4 SigLIP frame embeddings, cached 1 frame/s SigLIP + CLIP baselines in `search/index/`), methods `captions` / `siglip_selected` / `hybrid` (RRF) / `siglip_1fps` / `clip_1fps`, `make_queries` (UCA sentences; one query per MEVA activity via `MEVA_QUERIES`), `evaluate` / `metrics` (R@K at IoU 0.3/0.5 on a `moment_s` window, lenient, MRR, reachable %), `latency_probe`, `result_strip` for explore section 8. Notebook committed **without** outputs |
| `video_utils.py`, `visualize.py` | Frame reading; plots, GIFs, tracker-box drawing |
| `docs/pipeline.md` | Method write-up with references and the latest batch results; keep it in sync when a step or default changes |
| `docs/explanation.md` | Step-by-step tuning guide: every setting, its default, what raising/lowering it does, what ran / was skipped in the last run |
| `video/`, `outputs/` | Git-ignored. Videos and results stay local. Datasets live outside the repo (`datasets.DATA_ROOT`) |

The slow decode pass is cached in `outputs/batch/<dataset>/<video>/cache/scan_<hash>.npz`. Changing a scan parameter
invalidates it; changing a segment or keyframe parameter does not.

## `gpu_stages.py` and the v1/v2 merge

v2 was a standalone 1,600-line script that re-implemented diff + MOG2. In the merge its stage 1, CLI and duplicate
helpers were removed (the original is still on `main` as `rewind_pipeline_Muzz.py`); what's left is only what needs
deep models. Taken over from v2 into v1's modules: the quieter-half threshold (`q_noise`), colour MOG2, the fused
vote, ground-truth evaluation.

- Config: `gpu_stages.DEFAULT_CONFIG` (`deep_gate`, `segments.heartbeat_s`, `events`, `sampling`, `runtime`).
- Hand-off: `outputs/<video>/events/index_records.jsonl` + `index_embeddings.npy` (SigLIP, 768-d for
  `siglip-base-patch16-224`), one record per keyframe of the primary sampler (`kcenter`, falls back to `phash`).
- A model that fails to load is skipped with a warning (`models.failed`). SigLIP needs `sentencepiece`.
- Stage 5 (`captions.py`) reads that hand-off (or stage 2's `keyframes.csv`) and writes `captions/captions.csv`.
  Unlike `Models`, a VLM that fails to load is an error per video (`error_vlm_<level>.json`), not a silent skip. The
  detector's events are stored next to captions but deliberately not put in the prompt (independent check).
- Known weak spots (café video): track IDs switch in crowds, `crowd`/`loitering` fire constantly in busy scenes,
  X-CLIP's softmax over 13 prompts rarely passes `xclip_threshold` 0.35, the deep gate's first frames become its
  background (people there at the start count as change for ~1 min).
