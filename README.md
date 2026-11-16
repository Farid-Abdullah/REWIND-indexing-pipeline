# Motion gating and keyframe extraction

**Goal:** in a long video from a fixed camera, find *when something moves* and pick *a few frames that show it*.
An expensive model (e.g. one that describes images) then only looks at those frames instead of all 54,000.
On the 30-min test video: **109 keyframes instead of 1,801** (1 per second), so 16.5× fewer.

## Run it

```bash
conda env create -f environment.yml          # once
conda activate motion-gating
python -m ipykernel install --user --name motion-gating   # once, makes the env available in Jupyter(vscode prompts you for env automatically)

python meva_30min.py        # download the test video into video/ (skip if you already have it)
python motion_gating.py     # stage 1  (or: python motion_gating.py video/other.mp4, meva_30min.mp4 is defualt)
python keyframes.py         # stage 2
```

Or open `explore.ipynb` (kernel **motion-gating**) and *Run All*: it runs both stages and shows the results.
Results go to `outputs/<video name>/`.

## Files

| File | What it does |
|---|---|
| `meva_30min.py` | Downloads six 5-min clips from the public MEVA dataset and joins them into `video/meva_30min.mp4` |
| `motion_gating.py` | **Stage 1.** Scores how much moves in each frame and finds the time ranges ("segments") with motion |
| `keyframes.py` | **Stage 2.** Picks a few representative frames per segment and saves them as JPGs |
| `video_utils.py` | Reads frames from a video (used by both stages) |
| `visualize.py` | Thumbnails, plots and GIFs for the notebook |
| `explore.ipynb` | Runs both stages and shows timelines, examples, all keyframes and checks |

## Stage 1: motion gating (`motion_gating.py`)

1. **Look at 5 frames per second, not 30.** Neighbouring frames are almost identical, so this is 6× less work.
2. **Shrink to 320 px, greyscale, blur.** Motion doesn't need colour or detail. This is faster, and blurring hides pixel noise.
3. **Detect moving pixels, two ways** (compared in the notebook):
   - *Frame differencing*: a pixel moved if its brightness changed a lot since the previous frame. Simple, but only sees things moving right now.
   - *MOG2 background subtraction*: learns what each pixel normally looks like over the last ~100 s. Anything that doesn't fit is "foreground". Used by default.
4. **Clean up.** *Morphological opening*, a clean-up filter for black and white masks to remove small specks of noise but keeps real blobs(people, cars). This is to not allow scattered flickering to add into the "% of pixels moving" score.
5. **Lighting filter (MOG2).** Cloud shadows look like motion to MOG2. But a shadowed patch keeps its texture, so blobs that still
   *correlate* with the background (NCC > 0.7) are dropped. Without this, the strongest "motion" in the video was sunlight on the lawn.
6. **Score** = % of pixels moving.
7. **Threshold** = median + k × MAD. *MAD* is a spread measure that ignores spikes. So this means "k times more than normal", adapted to each video.
8. **Segments.** *Hysteresis*: start above a high threshold, stop below a lower one, so it doesn't flicker on and off.
   Then merge gaps under 3 s, drop pieces under 1 s, and add 2 s padding on each side.

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

## Outputs (`outputs/<video name>/`)

| File | Contents |
|---|---|
| `segments.csv` | one row per segment: start, end, duration, peak motion |
| `keyframes/` + `keyframes.csv` | the JPGs (`kf_00-12-34.500_seg007.jpg`) and one row each: time, segment, reason |
| `motion_scores.csv` | both motion scores for every analysed frame |

## Main knobs

| Parameter | Where | Effect |
|---|---|---|
| `k` | `SEGMENT_PARAMS` | higher = only stronger motion counts, fewer segments |
| `method` | `SEGMENT_PARAMS` | `"mog2"` or `"diff"` |
| `min_gap_s` | `KEYFRAME_PARAMS` | the main control of how many keyframes you get (3 s → 109, 10 s → 56) |
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
