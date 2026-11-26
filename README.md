# REWIND: motion gating, keyframes, events, captions, search

**Goal:** in a long video from a fixed camera, find *when something happens* and pick *a few frames that show it*.
Expensive models (captioning, the search index) then only look at those frames instead of every frame.
On the 30-min MEVA test video the v1 settings kept 109 keyframes instead of 1,801 (1 per second), 16.5× fewer.

**Docs:** `docs/explanation.md` walks through every step and setting (for tuning) and lists what was skipped or
failed in the last run; `docs/pipeline.md` is the method write-up with references and results.

## Layout

```
notebooks/            run these, in order
  1_pipeline.ipynb    videos -> segments, keyframes, events, SigLIP embeddings, captions (outputs/batch/)
  2_search.ipynb      search index, test queries from the ground truth, scores vs baselines (outputs/batch/search/)
  3_explore.ipynb     look at everything (no re-running)
rewind/               the code (imported by the notebooks as `from rewind import ...`)
  paths.py            where outputs/, selection.csv and the datasets are
  data.py             MEVA / UCA discovery, video selection, ground truth
  motion_gating.py    stage 1      keyframes.py   stage 2
  gpu_stages.py       stages 3-4 (YOLO-World + ByteTrack, SigLIP, X-CLIP, frame selection, hand-off)
  captions.py         stage 5 (Qwen-VL clip and frame captions)
  batch.py            runs the phases over every video (what 1_pipeline.ipynb calls)
  search.py           search + its evaluation        caption_eval.py   caption quality checks
  utils/              video_utils.py (frame reading), visualize.py (plots, GIFs, boxes)
docs/                 pipeline.md (method + results), explanation.md (every setting, for tuning)
runs/2026-11-26/      the notebooks of the 26 Nov run with all their outputs
outputs/              results (git-ignored)    selection.csv   the videos of the batch (made on the first run)
```

## Run it

**Once per machine.** Clone, then register the package in the environment the notebooks run in (on the VM: the
**rewind (5T)** kernel's environment). `pip install -e .` installs nothing else and never touches torch; if you
skip it, the notebooks still find the code as long as they are opened from `notebooks/`.

```bash
cd /mnt/5T_storage
git clone https://github.com/Farid-Abdullah/REWIND-indexing-pipeline rewind
cd rewind
pip install -e .              # inside the environment of the notebook kernel
```

The repository was called `cctv-motion-gating` until 26 Nov 2026. GitHub redirects the old URL, but an existing
checkout should point at the new one: `git remote set-url origin https://github.com/Farid-Abdullah/REWIND-indexing-pipeline`.

An existing checkout: `git pull`, then `pip install -e .` once. `outputs/` and `selection.csv` stay where they
were (the repository folder), so earlier results are still found.

**Three notebooks, always in this order (all in `notebooks/`):**

1. **`1_pipeline.ipynb`**: check the config cell (`VLM_MODEL`, `VLM_DEVICE`, `STAGES`), then *Run All*. It checks
   the environment first (GPU, `sentencepiece`, YOLO, the Qwen folder) and stops before touching anything if
   something is missing; then deletes the old results (`FRESH_RUN = True`) and runs every phase over every video.
   No pictures, one line per video; everything also goes to `outputs/batch/run_log.txt`. About 45 min for 2 h of video.
2. **`2_search.ipynb`**: *Run All*. Builds the search index from the captions and SigLIP embeddings, makes the test
   queries from the ground truth, runs them through every search method and the two baselines, scores them, saves
   `outputs/batch/search/`. Ends with a *try a query* cell. A few minutes.
3. **`3_explore.ipynb`**: *Run All* to see all the results: gating, keyframes, events, the captions next to their
   frames and the labelled moments, and (section 8) the search scores and example results with their frames.

If the run stops part-way (VS Code disconnects, you interrupt it): set `FRESH_RUN = False` in `1_pipeline.ipynb`
and *Run All* again; it continues where it stopped. The same functions also run from a terminal
(`python -m rewind.batch --fresh`, see its docstring), but the notebooks are the main way.

- **Which videos:** `selection.csv`, made on the first run: ~1 h of MEVA (whole 5-min time slots of all 4 cameras,
  annotated slots first) + ~1 h of UCA `dev_videos` (spread over categories, test split first). Edit it to change
  the set, or delete it to pick again. Data paths: `DATA_ROOT` in the config cell (default in `rewind/paths.py`:
  `/mnt/5T_storage/clipsExtraction/datasets`).
- **Four phases, in order, each over every video:** `cpu` (stages 1-2), `gpu` (stages 3-4), `vlm_clips` (a caption
  per clip), `vlm_frames` (a caption per selected frame; the slow one, hours). A later phase never costs earlier
  results; `3_explore.ipynb` works with whatever has finished.
- **Resume:** with `FRESH_RUN = False`, `cpu` / `gpu` skip finished videos and the caption phases continue at the
  next clip / frame. A failing video is logged and skipped; its traceback is in
  `outputs/batch/<dataset>/<video>/error_<phase>.json`.
- **Ground truth** comes from the datasets themselves (`rewind/data.py`): MEVA activity labels (`*.activities.yml`,
  frame numbers / fps) and UCA sentences with timestamps. No hand-made CSV needed.
- **Plumbing check without GPU or data:** `REWIND_MOCK=1` (fake models), with `REWIND_DATA_ROOT`, `REWIND_OUT_ROOT`
  and `REWIND_SELECTION` pointing at a small fake dataset.

**Single video / command line:** `python -m rewind.motion_gating video/x.mp4`, then
`python -m rewind.keyframes video/x.mp4` (results in `outputs/<video name>/`; `video/` is git-ignored).

**Environment** (if you need a new one):

```bash
# full pipeline (GPU): any Python >= 3.11 environment; install torch for the machine's CUDA first
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name rewind   # once, makes the env available in Jupyter

# local, CPU only (stages 1-2 and the notebooks), e.g. on Windows
conda env create -f environment.yml && conda activate motion-gating && pip install -e .
```

## Stage 1: motion gating (`rewind/motion_gating.py`)

1. **Look at 5 frames per second, not 30.** Neighbouring frames are almost identical, so this is 6× less work.
2. **Shrink to 320 px and blur.** Motion doesn't need detail. This is faster, and blurring hides pixel noise.
   Frame differencing works on greyscale; MOG2 keeps the colour (see below).
3. **Detect moving pixels, two ways** (plus the optional deep gate and a fused vote, see below):
   - *Frame differencing*: a pixel moved if its brightness changed a lot since the previous frame. Simple, but only sees things moving right now.
   - *MOG2 background subtraction*: learns what each pixel normally looks like over the last ~100 s. Anything that doesn't fit is "foreground". Used by default.
     Runs on the **colour** frame (`mog2_color`): MOG2 drops pixels it thinks are shadows (darker, same colour), and on a
     greyscale frame that includes people in mid-dark clothes on a light floor.
4. **Clean up.** *Morphological opening*, a clean-up filter for black and white masks to remove small specks of noise but keeps real blobs(people, cars). This is to not allow scattered flickering to add into the "% of pixels moving" score.
5. **Lighting filter (MOG2).** Cloud shadows look like motion to MOG2. But a shadowed patch keeps its texture, so blobs that still
   *correlate* with the background (NCC > 0.7) are dropped. Without this, the strongest "motion" in the video was sunlight on the lawn.
6. **Score** = % of pixels moving.
7. **Threshold** = median + k × MAD. *MAD* is a spread measure that ignores spikes. So this means "k times more than normal", adapted to each video.
   "Normal" is measured on the **quietest half** of the samples (`q_noise=0.5`): on a busy camera the overall median is
   already motion, and the threshold would sit above real activity (it did on the café video). `q_noise=1.0` = all samples.
8. **Segments.** *Hysteresis*: start above a high threshold, stop below a lower one, so it doesn't flicker on and off.
   Then merge gaps under 3 s, drop pieces under 1 s, and add 2 s padding on each side.

**Optional gates** (config cell of `1_pipeline.ipynb`): the *deep gate* (`gpu_stages.scan_deep`) compares DINOv2 features of 14×14-pixel patches
with a slowly learned background, 2 frames/s, GPU. The *fused gate* divides each gate's score by its own threshold and
averages the two highest: a soft "two gates agree" vote. On the café ground truth MOG2 and fused were nearly tied, so
MOG2 stays the default (`GATE` in the config).

The slow part (decoding, ~50 s) is **cached** in `outputs/<video>/cache/` after first run, so changing thresholds afterwards takes about a second.

## Stage 2: keyframes (`rewind/keyframes.py`)

For each segment, keep:
- **first**: the first frame
- **peak**: the frame with the most motion
- **change**: a frame whenever the scene changed enough since the last keyframe (at least 3 s apart)
- **maxgap**: a forced frame if 30 s pass without one

Then drop **near-duplicates**. Outside segments, keep one **heartbeat** frame every 5 min, so slow changes aren't missed.

*How "changed enough" is measured:* a **perceptual hash** (pHash) turns an image into a 64-bit fingerprint of its rough shapes, and
similar images get similar bits. The frame is split into 4×4 tiles with one hash each, so a few small people moving still change their
tile. Distance = how many bits differ (*Hamming distance*).

## Stages 3-4: events and frames for the index (`rewind/gpu_stages.py`, GPU)

Only inside segments, 4 frames/s:
- **tracks**: YOLO-World finds people and objects, ByteTrack follows them; rules turn tracks into events
  (appears, loitering, running, object left behind, crowd)
- **SigLIP**: compares each frame with text prompts ("a person carrying a box"...). Its image embeddings are also
  what the search index stores
- **X-CLIP**: the same for 8-frame clips, so it can see motion over time (falling, running)

Then several **samplers** choose which of those frames go to the captioner and the index (`kcenter` by default),
scored side by side with stage 2's keyframes. The chosen frames are written to `events/index_records.jsonl`
(camera, time, events, objects) + `events/index_embeddings.npy` (SigLIP, 768 numbers per frame).

## Stage 5: captions (`rewind/captions.py`, GPU)

A vision-language model (Qwen2.5-VL-7B by default; set `CFG["vlm"]["model"]` to another name or a local folder,
Qwen3-VL works too) writes a JSON description of:
- **every selected frame** (stage 4's `index_records.jsonl`, or stage 2's keyframes if stage 4 hasn't run):
  caption, scene, lighting, people (clothes, action, where), vehicles, objects, anything unusual, what's unclear;
- **every clip** (a motion segment cut into pieces of at most 60 s): up to 6 of its frames in time order, so it can
  describe what *happens* (enters, picks up, leaves). A clip has a start and an end: it's what a search returns.

It is told to describe only what it can see. Everything goes to `captions/captions.csv` (one row per frame or clip:
camera, start/end, clock time, text, `search_text`, preview image). Clips are captioned for every video first
(phase `vlm_clips`, ~7 s per clip on one L4), then frames (phase `vlm_frames`, ~5-10 s per frame), so an
interrupted run still leaves every video searchable by clip. The model is loaded once for both.

**Caption quality** (`rewind/caption_eval.py`, shown in `3_explore.ipynb` section 6, no re-run needed):
- *detector cross-check* (automatic, every clip): what a caption mentions (people, vehicles, bags, ...) vs what
  YOLO-World tracked in the same seconds: likely hallucinations, likely misses, people-count agreement;
- *labels* (automatic): each UCA sentence / MEVA activity vs the caption of its clip, with a sentence-embedding
  model (`bge-small`), against other videos' captions as a baseline; plus top-1 / top-5 caption-only search;
- *hand check*: 30 random clips split over 5 people, `outputs/batch/clip_review/review_<k>.html` (frames next to
  the caption) + `review_<k>.csv` to fill in; gives the hallucination and miss rate, and how far to trust the
  detector cross-check. Model: `VLM_MODEL` in `1_pipeline.ipynb` (the VM's local Qwen2.5-VL-7B folder); needs `torch`,
`transformers>=4.49` and ~17 GB free on each GPU in `VLM_DEVICE`. `VLM_DEVICE = ["cuda:0", "cuda:1"]` (the default)
runs one copy of the model per L4, each on about half the video-time, so captioning takes about half as long;
`"cuda:0"` uses one GPU. Plumbing check without a GPU: `REWIND_MOCK=1`.

## Search (`rewind/search.py`, `notebooks/2_search.ipynb`)

Type a sentence, get ranked moments (video, camera, start-end, best frame). Everything is in memory: no database,
no server. Methods, all scored the same way:

| Method | The query is compared with | Returns |
|---|---|---|
| `captions` | each clip's caption (`bge-small` sentence embeddings, with bge's retrieval prefix on the query) | clips |
| `siglip_selected` | the SigLIP embeddings of stage 4's selected frames (SigLIP's text tower; one result per 10 s) | frames, mapped to their clip |
| `hybrid` (default) | both, merged by reciprocal rank fusion (`rrf_k = 60`); a clip's SigLIP score = its best frame | clips |
| `siglip_1fps` | baseline: SigLIP on 1 frame/s of every whole video (no gating, no captions) | frames |
| `clip_1fps` | baseline: the same with CLIP ViT-B/32 | frames |

Each result has two spans: the **clip** (up to 60 s) and a **window** of `moment_s` = 10 s around its best frame,
because a 60-s clip can never reach IoU 0.3 with a 5-s event.

**Test queries** (`outputs/batch/search/queries.csv`, written from the ground truth on the first run; edit it or add
rows): every UCA sentence of the selected UCA videos (answer: its own span) and one operator-style sentence per MEVA
activity type, e.g. `person_opens_vehicle_door` → *"a person opens a car door"* (answers: every labelled instance on
any camera; the mapping is `search.MEVA_QUERIES`). Plus 20 free-text operator queries without ground truth, judged by eye.

**Scores** (`metrics.csv`) per dataset, method and setting (`corpus` = all videos of the dataset, `video` = only the
query's own video, UCA only): Recall@1/5/10 at temporal IoU 0.3 and 0.5 (window span), lenient Recall@K (the clip
overlaps an answer), MRR, **reachable %** (an answer overlaps some clip: the most the gated methods can reach) and
lenient R@5 over reachable queries only (is the search itself good?), plus ms per query.

Outputs in `outputs/batch/search/`: `queries.csv`, `results.csv` (top 10 per query x method x setting),
`per_query.csv` (first correct rank per query), `metrics.csv`, `latency.csv`, `demo_results.csv`, `search_run.json`,
and `index/` (the cached baseline embeddings; delete it to re-embed). Plumbing check without models: `REWIND_MOCK=1`.

Known weak spots (café video): track IDs switch in crowds, so one person can "appear" many times; `crowd` and
`loitering` fire almost constantly in a busy lobby; X-CLIP rarely passes its threshold. The rules were written for
outdoor CCTV and need tuning per scene.

## Outputs (`outputs/batch/<dataset>/<video name>/`)

Batch-level files in `outputs/batch/`: `batch_summary.csv` (one row per video, every number below), `selection_used.csv`,
`batch_config.json`, `run_log.txt` (everything the run printed, with times), `clip_review/` (hand-check sheets,
made by `3_explore.ipynb`), `search/` (search scores, see above). Per video, `summary_cpu.json` / `summary_gpu.json` hold that video's row, and `gt.csv` the
labelled items with a `has_keyframe` column.


| File | Contents |
|---|---|
| `segments.csv` | one row per segment: start, end, duration, peak motion |
| `keyframes/` + `keyframes.csv` | the JPGs (`kf_00-12-34.500_seg007.jpg`) and one row each: time, segment, reason |
| `motion_scores.csv` | every gate's score for every analysed frame |
| `gt.csv`, `gt_eval_gates.csv`, `gt_eval_keyframes.csv` | ground truth and how each gate / keyframe set did on it, incl. **uniform sampling with the same number of frames** and 1 frame/s as baselines (only for videos with annotations) |
| `cache/` | the slow decode pass (and the deep gate), so re-runs are fast. Safe to delete |
| `captions/` | stage 5: `captions.csv` (frames + clips in one table), `frame_captions.jsonl`, `clip_captions.jsonl`, `frames/` (what the model saw) |
| `events/` | stages 3-4 (GPU): `events.csv` (merged events), `tracks.csv` + `tracks_raw.csv`, `xclip_windows.csv`, `candidate_frames.csv`, `sampling_comparison.csv`, `keyframes_<sampler>.csv` (+ JPGs in `keyframes/kcenter/`), `gifs/`, `timeline.png`, and the hand-off **`index_records.jsonl` + `index_embeddings.npy`** |

## Main knobs

| Parameter | Where | Effect |
|---|---|---|
| `k` | `SEGMENT_PARAMS` | higher = only stronger motion counts, fewer segments |
| `q_noise` | `SEGMENT_PARAMS` | share of quietest samples that defines "normal" (0.5; 1.0 = all samples) |
| `mog2_color` | `SCAN_PARAMS` | MOG2 on colour (True) or greyscale (False) |
| `GATE` | notebook config (`method` in `SEGMENT_PARAMS` for the command line) | which gate feeds the later stages: `"mog2"` (default), `"diff"`, `"fused"`, `"deep"` |
| `min_gap_s` | `KEYFRAME_PARAMS` | the main control of how many keyframes you get |
| `change_threshold` | `KEYFRAME_PARAMS` | higher = a bigger scene change is needed for a new keyframe |
| `heartbeat_s` | `KEYFRAME_PARAMS` | how often to sample quiet periods |
| `ignore_polygons` | `SCAN_PARAMS` | regions to ignore (e.g. sky) |

Note: "inactive" means *no significant motion*, not *no people*. People standing still blend into MOG2's background after ~100 s.

## Syncing with a shared VM (proposal, not set up yet)

**Problem:** a copy of this repo runs on a shared VM. Everyone logs in with the same username, so putting
anyone's GitHub credentials there (a token, an SSH key, or SSH agent forwarding) would let every user of that
account push as that person.

**Idea:** a laptop that already has GitHub access is the go-between. It treats the VM as a second git remote
over plain SSH. No GitHub credentials ever go on the VM.

```
GitHub  ⇄  laptop  ⇄  VM
```

One-time setup (replace `<user>`, `<vm-host>` and `<path>`):

```bash
# on the VM, inside the repo
git -c user.name="Your Name" -c user.email="you@example.com" commit -am "VM changes"  # commit VM work without changing the shared git identity
git config receive.denyCurrentBranch updateInstead   # let pushes update the VM's checked-out files (this repo only)

# on the laptop
git remote add vm ssh://<user>@<vm-host>/<path>
# optional: copy the laptop's *public* key to the VM's ~/.ssh/authorized_keys to skip password prompts
```

Each sync, from the laptop:

```bash
git fetch vm && git merge vm/main   # VM -> laptop
git push origin main                # laptop -> GitHub
git push vm main                    # laptop -> VM
```

- `git push vm` is refused while the VM has uncommitted edits, so it can't overwrite someone's work in progress.
  Commit or stash on the VM first.
- `video/`, `outputs/` and `selection.csv` are git-ignored, so videos and results don't sync. Copy them with `scp` or `rsync`
  if needed.
- Any teammate can be the go-between. They just add the `vm` remote on their own machine.
