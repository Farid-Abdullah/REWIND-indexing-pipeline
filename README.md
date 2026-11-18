# REWIND ingest: motion gating, keyframes, events

**Goal:** in a long video from a fixed camera, find *when something happens* and pick *a few frames that show it*.
Expensive models (captioning, the search index) then only look at those frames instead of every frame.
On the 30-min MEVA test video the v1 settings kept 109 keyframes instead of 1,801 (1 per second), 16.5× fewer.

## Run it

```bash
conda env create -f environment.yml          # once (on the VM: add -p /mnt/5T_storage/envs/motion-gating)
conda activate motion-gating
python -m ipykernel install --user --name motion-gating   # once, makes the env available in Jupyter
python meva_30min.py                          # optional: download a 30-min test video into video/
```

Open **`pipeline.ipynb`** (kernel **motion-gating**), set `VIDEO` in the config cell, *Run All*.
Results go to `outputs/<video name>/`. The two CPU stages also run from the command line:
`python motion_gating.py video/x.mp4` then `python keyframes.py video/x.mp4`.

**GPU stages** (deep gate, events, frame selection for the index) need extra packages:

```bash
export PIP_CACHE_DIR=/mnt/5T_storage/pip_cache TMPDIR=/mnt/5T_storage/tmp   # on the VM: keep pip off the root disk
pip install torch transformers ultralytics lap sentencepiece protobuf
python -c "import cv2, torch; print(cv2.__version__, torch.cuda.is_available())"
```

`ultralytics` installs `opencv-python` next to our `opencv-python-headless`. If `import cv2` breaks afterwards:
`pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python-headless`.
Any model that fails to load is skipped with a warning, so the notebook still runs without a GPU, with fewer sections.
`"mock_models": True` in the config cell runs everything with fake models: a quick plumbing check, numbers meaningless.

## Files

| File | What it does |
|---|---|
| `pipeline.ipynb` | **The demo.** Runs every stage on one video and shows timelines, examples, checks and comparisons |
| `motion_gating.py` | **Stage 1.** Scores how much moves in each frame, finds the time ranges ("segments") with motion. Also the ground-truth checks |
| `keyframes.py` | **Stage 2.** Picks a few representative frames per segment and saves them as JPGs |
| `gpu_stages.py` | Deep-model parts (from Muzayyan's v2 pipeline): DINOv2 motion gate, **stage 3** events (YOLO-World + ByteTrack, SigLIP, X-CLIP), **stage 4** frame selection and the `index_records.jsonl` hand-off |
| `video_utils.py`, `visualize.py` | Reading frames; thumbnails, plots and GIFs for the notebook |
| `meva_30min.py` | Downloads six 5-min clips from the public MEVA dataset and joins them into `video/meva_30min.mp4` |
| `environment.yml` | The conda environment (CPU part) |

## Stage 1: motion gating (`motion_gating.py`)

1. **Look at 5 frames per second, not 30.** Neighbouring frames are almost identical, so this is 6× less work.
2. **Shrink to 320 px, greyscale, blur.** Motion doesn't need colour or detail. This is faster, and blurring hides pixel noise.
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

**Optional gates** (`pipeline.ipynb`): the *deep gate* (`gpu_stages.scan_deep`) compares DINOv2 features of 14×14-pixel patches
with a slowly learned background, 2 frames/s, GPU. The *fused gate* divides each gate's score by its own threshold and
averages the two highest: a soft "two gates agree" vote. On the café ground truth MOG2 and fused were nearly tied, so
MOG2 stays the default (`GATE` in the config).

The slow part (decoding, ~50 s) is **cached** in `outputs/<video>/cache/` after first run, so changing thresholds afterwards takes about a second.

## Stage 2: keyframes (`keyframes.py`)

For each segment, keep:
- **first**: the first frame
- **peak**: the frame with the most motion
- **change**: a frame whenever the scene changed enough since the last keyframe (at least 3 s apart)
- **maxgap**: a forced frame if 30 s pass without one

Then drop **near-duplicates**. Outside segments, keep one **heartbeat** frame every 5 min, so slow changes aren't missed.

*How "changed enough" is measured:* a **perceptual hash** (pHash) turns an image into a 64-bit fingerprint of its rough shapes, and
similar images get similar bits. The frame is split into 4×4 tiles with one hash each, so a few small people moving still change their
tile. Distance = how many bits differ (*Hamming distance*).

## Stages 3-4: events and frames for the index (`gpu_stages.py`, GPU)

Only inside segments, 4 frames/s:
- **tracks**: YOLO-World finds people and objects, ByteTrack follows them; rules turn tracks into events
  (appears, loitering, running, object left behind, crowd)
- **SigLIP**: compares each frame with text prompts ("a person carrying a box"...). Its image embeddings are also
  what the search index stores
- **X-CLIP**: the same for 8-frame clips, so it can see motion over time (falling, running)

Then several **samplers** choose which of those frames go to the captioner and the index (`kcenter` by default),
scored side by side with stage 2's keyframes. The chosen frames are written to `events/index_records.jsonl`
(camera, time, events, objects) + `events/index_embeddings.npy` (SigLIP, 768 numbers per frame).

Known weak spots (café video): track IDs switch in crowds, so one person can "appear" many times; `crowd` and
`loitering` fire almost constantly in a busy lobby; X-CLIP rarely passes its threshold. The rules were written for
outdoor CCTV and need tuning per scene.

## Outputs (`outputs/<video name>/`)

| File | Contents |
|---|---|
| `segments.csv` | one row per segment: start, end, duration, peak motion |
| `keyframes/` + `keyframes.csv` | the JPGs (`kf_00-12-34.500_seg007.jpg`) and one row each: time, segment, reason |
| `motion_scores.csv` | every gate's score for every analysed frame |
| `gt_eval_*.csv` | ground-truth results (if `GT_CSV` is set) |
| `events/` | stages 3-4: `events.csv`, `tracks.csv`, `sampling_comparison.csv`, `keyframes_<sampler>.csv`, and the hand-off `index_records.jsonl` + `index_embeddings.npy` |

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
- `video/` and `outputs/` are git-ignored, so videos and results don't sync. Copy them with `scp` or `rsync`
  if needed.
- Any teammate can be the go-between. They just add the `vm` remote on their own machine.
