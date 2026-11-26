# What the pipeline does, step by step (and what to tune)

Every step that `pipeline.ipynb` runs, in order, with every setting, what it does, what happens if you raise or
lower it, and what it did in our first real run. Written so we can tune the pipeline on purpose instead of by
guesswork. The method write-up with references is `docs/pipeline.md`; this file is the hands-on companion.

**The run this file describes:** 19 Nov 2026, branch `moreVids`, commit `ea088de`, on the team VM (one L4 GPU
used). 38 videos, 2.00 h of video: 12 MEVA clips (1.0 h) + 26 UCA videos (1.0 h). All default settings.
Wall time 15.3 min (CPU phase 7.9 min, GPU phase 7.4 min). No video failed.
Kernel "rewind (5T)" ran **Python 3.13** from `/mnt/5T_storage/clipsExtraction/tools/awsenv`, not the
`motion-gating` conda env (3.11) described in `environment.yml`. Packages must be installed into that env.

Conventions: `NAME=value` is a setting and its default. "Config" means the config cell of `pipeline.ipynb`
(`CFG`), which overrides the code defaults. Code is referred to as `file.function`.

---

## At a glance: what ran, what was skipped, what failed

| Step | Status in the 19 Nov run | Why / what it means |
|---|---|---|
| 0. Find videos, pick ~1 h per dataset, load labels | ✅ ran | 12 MEVA clips (all annotated), 26 UCA videos (all annotated) |
| 1. Motion gating: frame differencing | ✅ ran | computed for every video, compared, but **not** the gate used downstream |
| 1. Motion gating: MOG2 | ✅ ran, **used** | `gate="mog2"` feeds keyframes and stages 3-4 |
| 1. Motion gating: fused vote | ✅ ran | average of diff and MOG2 (only 2 gates, so not a real "2 of N" vote) |
| 1. Motion gating: deep gate (DINOv2) | ⏭️ **off** | `deep_gate=False` by choice (a full-video GPU pass; earlier results didn't justify it) |
| 2. Keyframes | ✅ ran | 1,037 keyframes for 2 h; JPGs saved for the MOG2 gate only |
| Ground-truth check | ✅ ran | all 38 videos have labels |
| 3. Detector + tracker (YOLO-World + ByteTrack) | ✅ ran | most events are `person_appears`; IDs switch a lot |
| 3. SigLIP tags + **image embeddings** | ❌ **failed to load** | `SiglipTokenizer requires the SentencePiece library`. Fix: `pip install sentencepiece protobuf` in the awsenv, restart the kernel |
| 3. X-CLIP clip tags | ✅ loaded and ran | in the video inspected in `explore.ipynb`, every event key is one the tracker rules produce; X-CLIP contributed nothing visible (check the `source` column of `events.csv`) |
| 4. Samplers `novelty`, `kcenter`, `importance` | ⏭️ **skipped** | they need SigLIP embeddings |
| 4. Primary sampler | ⚠️ **fell back** | `primary="kcenter"` didn't run, so the hand-off used `phash` |
| 4. Stage-4 JPGs | ⚠️ **none written** | `export_jpgs=["kcenter"]` and kcenter didn't run |
| 4. Hand-off `index_records.jsonl` | ⚠️ written, **without embeddings** | `index_embeddings.npy` is empty; `embedding_row` points nowhere; `siglip_tag` is empty |
| Coverage / redundancy metrics | ⚠️ NaN | need embeddings |
| 5. Captions (VLM) | ⏭️ **not in this run** | `captions.py` was added on 26 Nov; tested only with a fake captioner. Run `STAGES=["vlm"]` |

So after this run we have segments, keyframes, tracks and a ground-truth check, but **no search embeddings yet**.
Installing `sentencepiece` and re-running only the GPU phase (`STAGES=["gpu"]`, `FORCE=True`) fixes most of the
⚠️ and ❌ rows.

```
video (25-30 frames/s)
 │
 ├─ 0  datasets.py     which videos, their camera / clock time / labels
 ├─ 1  motion gating   5 samples/s, 320 px, CPU      → motion score per sample → threshold → segments
 ├─ 2  keyframes       inside segments, CPU          → a few frames per segment (+ 1 per 5 idle min)
 ├─ GT check           CPU                           → did segments / keyframes catch the labelled moments?
 ├─ 3  events          inside segments, 4 frames/s   → tracks, tags, events, importance per frame
 ├─ 4  frame selection same candidates               → frames for the captioner / index + hand-off files
 └─ 5  captions        stage 4's frames + clips       → structured text per frame and per clip (captions.csv)
```

---

## Step 0: datasets and selection (`datasets.py`)

### What is on disk (section 1 of `pipeline.ipynb`)

| | Videos | Hours | With labels | Labelled items |
|---|---|---|---|---|
| MEVA (`meva/videos`) | 82 | 5.74 | 61 (exact name match), 21 none | 2,923 activities |
| UCA (`uca/dev_videos`) | 166 | 9.01 | 166 | 1,538 sentences |

- MEVA: 4 cameras at one school site on 2018-03-13 (G421 30 files, G328 18, G424 17, G336 17). 20 files are
  shorter than 4.5 min: G421 is split into fragments.
- UCA: **all 166 dev videos are from the `val` split**, none from `test`. Good: the test split stays untouched
  for the final Recall@K numbers. It also means `prefer_split="test"` had no effect.
- Total on disk: **14.75 h**, not 17.

### How MEVA labels are read
Files `*.activities.yml` (KPF format), one activity per line:
`{'act': {'act2': {'person_picks_up_object': 1.0}, 'timespan': [{'tsr0': [2949, 2962]}], 'actors': [...]}}`.
`tsr0` is a **frame range**, so seconds = frame / fps. Files are matched to videos by name without the `.r13`
suffix; a ±5 s start-time match is the fallback (`gt_match="fuzzy"`; none were needed).

Only ~37 activity types are labelled (talking, picking up, opening a door...). Someone just walking past is
**not** labelled, so MEVA labels say "these moments must be caught", not "the rest is empty".

### How UCA labels are read
`UCFCrime_{Train,Val,Test}.json`: per video, a list of sentences with `[start, end]` seconds. The spans follow
each other back to back and cover the whole video, quiet parts included. So UCA can't tell us what is
"inactive"; it can tell us whether every described moment got a keyframe. (`revised_en_UCFCrime_Train.json` is
skipped: same videos, reworded.)

### Selection settings (config cell)

| Setting | Value | What it does |
|---|---|---|
| `MEVA_MINUTES` | 60 | MEVA minutes to pick |
| `select_meva(slot_min=5)` | 5 | clips are grouped into 5-min time slots; whole slots (every camera) are taken, so the set is multi-camera |
| `select_meva(prefer_gt=True)` | True | slots with more annotated clips come first, then by time |
| `MEVA_CAMERAS` | None | restrict to some cameras |
| `UCA_MINUTES` | 60 | UCA minutes to pick |
| `select_uca(prefer_split="test")` | "test" | within a category, test videos first (no effect here: all are val) |
| `select_uca(max_video_min=10)` | 10 | skip videos longer than 10 min, so one video can't eat the budget |
| `select_uca(seed=0)` | 0 | random order within a category (fixed, so the pick is repeatable) |
| `REMAKE_SELECTION` | False | `selection.csv` is reused if it exists; edit it by hand to change the set |

**What it picked:** MEVA slots 15:50, 15:55 and 16:05 × 4 cameras (the 16:00 slot, where G421 is fragmented, had
fewer annotated clips and was skipped). UCA: 2 videos from each of 13 categories, 9 s to 5.9 min long.

---

## Stage 1: motion gating (`motion_gating.py`)

**Job:** mark the time ranges where something moves. A miss here is fatal (no later stage sees that moment); a
false alarm only costs compute. So this stage should lean towards **recall**.

Three motion scores are computed in one decode pass (diff, MOG2, fused; the deep gate is off). Each is turned into
segments the same way (1.5, 1.6). One of them (`gate`) is used downstream.

### 1.1 Decoding and preprocessing (`scan_video`, `preprocess`) — shared by all methods

| Setting | Value | What it does | Raise it → | Lower it → |
|---|---|---|---|---|
| `sample_fps` | 5.0 | samples analysed per second of video (every 6th frame at 30 fps, every 5th at 25 fps) | catches shorter events, slower | faster; events under ~0.5 s can fall between samples |
| `width` | 320 | frames shrunk to 320 px wide (aspect kept) | sees smaller / farther people, slower, more noise | faster; small far-away people disappear |
| `blur_ksize` | 5 | 5×5 Gaussian blur before both methods | fewer noise pixels, small objects fade | more sensitive to noise and compression artefacts |
| `ignore_polygons` | [] | regions to ignore, in 0-1 coordinates (e.g. sky, a TV screen, a busy road) | — | — |

Every frame must still be **decoded** (a codec can't skip ahead), but only the samples are converted to images.
This pass is the slow part and is cached (`outputs/batch/<dataset>/<video>/cache/scan_<hash>.npz`); changing any
setting in this table re-decodes, changing anything in 1.5-1.6 or stage 2 doesn't.

The same pass also stores, per sample, the fingerprints stage 2 needs (`hash_grid=4` tile hashes,
`hist_bins=32` grey-level histogram) and the mean brightness.

**In the run:** decoding + scoring took 4.5 min per hour of MEVA (≈13× faster than real time) and 2.8 min per
hour of UCA (lower resolution).

### 1.2 Method A: frame differencing (`MotionScorer.update`) → `score_diff`

A pixel "moves" if its grey level changed by more than `diff_threshold` since the **previous sample** (0.2 s
earlier). Then a 3×3 opening removes isolated pixels. Score = share of pixels that moved.

| Setting | Value | What it does | Raise it → | Lower it → |
|---|---|---|---|---|
| `diff_threshold` | 15 | grey levels (of 255) a pixel must change | ignores noise and slow light changes, misses low-contrast people | catches dim / low-contrast motion, more noise |
| `morph_kernel` | 3 | size of the opening (erode then dilate) that removes specks | removes more specks, thin limbs vanish | keeps more noise |

Strength: no memory, so it reacts instantly and never "absorbs" anyone. Weakness: only sees what moves *right
now*; a person who stops disappears, and slow motion barely registers.

### 1.3 Method B: MOG2 background subtraction (`MotionScorer`) → `score_mog2` (**the gate used**)

For every pixel, OpenCV's MOG2 keeps a few Gaussian "usual values" learned from recent samples. A new value that
fits none of them is foreground. Then: shadows are dropped, a 3×3 opening cleans up, and the lighting filter
removes blobs that are only light changes.

| Setting | Value | What it does | Raise it → | Lower it → |
|---|---|---|---|---|
| `mog2_history` | 500 | samples remembered = **100 s** at 5 samples/s | people who stand still stay "foreground" longer; slower to adapt to light | adapts faster; still people vanish sooner |
| `mog2_var_threshold` | 16.0 | how far (squared std. deviations) a value may be from a Gaussian and still be "background" | fewer foreground pixels (misses low contrast) | more sensitive, more noise |
| `mog2_learning_rate` | -1 | -1 = automatic (≈ 1/history once warmed up; faster at the very start) | — | — |
| `mog2_color` | True | MOG2 runs on the colour frame, so "shadow" means *darker with the same colour* | — | False = greyscale: any pixel 50-100 % as bright as the background counts as shadow, so people in mid-dark clothes vanish |
| shadow rule | fixed | OpenCV marks shadows as 127; we keep only 255 | — | — |
| `morph_kernel` | 3 | as in 1.2 | | |

**Lighting filter** (`remove_illumination_blobs`), MOG2 only:

| Setting | Value | What it does | Raise it → | Lower it → |
|---|---|---|---|---|
| `mog2_illum_ncc` | 0.7 | a blob whose texture still correlates (NCC) with the background image above 0.7 is treated as a light change and removed. `None` = off | removes fewer blobs (more lighting false alarms) | removes more (risk: removes real people) |
| `min_pixels` (code) | 8 | blobs smaller than this are kept without judging | | |

**What the run showed (G421 15:55, `explore.ipynb` section 5):** the peak-frame mask is red on people across the
hall, as it should be, but some blobs on people near the bright window are **yellow** (removed by the lighting
filter). Flat or low-texture regions (plain clothes, a person in front of a plain wall) correlate easily with
the background, so the filter can eat real people. On synthetic test videos with flat-coloured moving blocks it
removed them entirely. Worth testing `mog2_illum_ncc=0.8` or `None` on MEVA.

**MOG2 restarts per clip.** Each 5-min MEVA clip is processed on its own, so MOG2 starts with no background and
learns fast for the first stretch of every clip. The first ~1-2 min of each clip are less reliable.

### 1.4 Method C: fused vote (`add_fused`) → `score_fused`; Method D: deep gate (off)

**Fused:** each gate's 1-s-smoothed score is divided by its own start threshold (1.0 = exactly at the
threshold); the fused score is the mean of the two highest ratios. With only diff and MOG2 it is simply their
average ratio. Its thresholds are fixed: start 1.0, stop `stop_ratio`=0.5. No settings of its own.

**Deep gate (DINOv2), `deep_gate=False`, not run.** Would add a third score from DINOv2 patch features compared
with a slowly learned background (`gpu_stages.DEFAULT_CONFIG["deep_gate"]`: `deep_fps=2`, `deep_z=3.0`,
`deep_min_dist=0.05`, `deep_alpha_bg=0.05`, `deep_alpha_fg=0.008`, `deep_warmup=10`). Needs a GPU pass over the
whole video. Earlier single-video tests: lowest recall of all gates, first minute unreliable.

### 1.5 From scores to a threshold (`robust_threshold`, `start_stop_thresholds`)

| Setting | Value | What it does | Raise it → | Lower it → |
|---|---|---|---|---|
| `threshold_mode` | "adaptive" | "adaptive" = computed per video; "fixed" = `fixed_threshold` | | |
| `k` | 3.0 | start threshold = median + k × 1.4826 × MAD of the quiet samples ("k standard deviations above normal") | fewer, shorter segments (lower recall) | more, longer segments (higher recall, less saving) |
| `q_noise` | 0.5 | median and MAD come from the quietest 50 % of samples only | 1.0 = all samples: a busy camera pushes the threshold above real activity | e.g. 0.2: "normal" taken from the quietest 20 %, lower threshold in busy scenes |
| `floor` | 0.001 per gate | the threshold never goes below 0.1 % of pixels | | |
| `fixed_threshold` | mog2 0.004, diff 0.006 | used only when `threshold_mode="fixed"` | | |
| `stop_ratio` | 0.5 | stop threshold = 0.5 × start (hysteresis) | segments end sooner | segments run on longer |

**MAD** = median of |x − median(x)|, a spread measure that a few big spikes (the motion we want) barely move.
×1.4826 makes it equal to the standard deviation for normally distributed data.

**This is where busy scenes break.** G421 15:55 (crowded indoor hall, ~25 people the whole time): the quieter
half of the samples already has 2-4 % of pixels moving, so the start threshold came out at **4.67 %** (stop
2.33 %). Long stretches with dozens of people walking (score ≈ 3 %) count as "inactive". The quieter-half rule
assumes at least half the time is empty; in this scene it never is. Options, in order of effort:
`q_noise=0.2`, a lower `k` (2.0), `threshold_mode="fixed"` with a value measured on an empty stretch of that
camera, or a per-camera threshold learned once from a quiet period.

### 1.6 From threshold to segments (`find_segments`)

Applied in this order:

| Step | Setting | Value | What it does | Raise it → | Lower it → |
|---|---|---|---|---|---|
| 1. smooth | `smooth_s` | 1.0 | rolling **median** over 1 s (5 samples) removes one-sample spikes | ignores short bursts (misses very short events) | more flicker |
| 2. hysteresis | (start / stop from 1.5) | | switch ON above start, OFF only below stop | | |
| 3. merge | `merge_gap_s` | 3.0 | join segments ≤ 3 s apart | fewer, longer segments | more, shorter segments |
| 4. drop | `min_duration_s` | 1.0 | drop segments shorter than 1 s | drops more short events | keeps blips |
| 5. pad | `pad_s` | 2.0 | add 2 s before and after (then merge overlaps) | safer start/end, more active time | tighter segments |

Output per segment: `segment_id, start_s, end_s, duration_s, peak_score, peak_time_s, peak_frame_idx, mean_score`
(`segments.csv`, chosen gate only). `motion_scores.csv` has every sample, every score and an `active_<gate>` flag.

### 1.7 Results per gate (both datasets, from `explore.ipynb` section 3)

| Dataset | Gate | Active % | Keyframes | Labelled items with a keyframe | Labelled events a segment covers |
|---|---|---|---|---|---|
| MEVA | diff | 44.3 | 313 | **51.0 %** | **69.3 %** |
| MEVA | fused | 43.0 | 296 | 51.2 % | 67.6 % |
| MEVA | **mog2 (used)** | 42.7 | 302 | 48.8 % | 63.8 % |
| UCA | diff | 74.0 | 864 | **88.1 %** | 83.5 % |
| UCA | fused | 75.0 | 818 | 85.6 % | **86.1 %** |
| UCA | **mog2 (used)** | 65.9 | 735 | 80.9 % | 75.8 % |

**Frame differencing beat MOG2 on both datasets**, at a similar share of active video on MEVA. That is the
opposite of what the single café video suggested. Switching `gate` to `"diff"` or `"fused"` is a one-word change
worth testing.

Per video, MEVA active share ranged from 12 % to 68 %, UCA from 19 % to 100 %. MEVA is **not** "hours of
nothing": this school site is busy most of the afternoon.

**Check this:** in G421 15:55 the score drops to exactly 0 for the last ~50 s while labelled activities continue
(red marks at the end of the timeline). A flat zero with a crowd present looks like frozen frames in the file
or a decoding problem, not empty footage. Open the video at 4:10-5:00 and check.

### Tuning cheat sheet for stage 1

| You want | Change |
|---|---|
| Fewer missed events (recall) | lower `k` (2-2.5), lower `q_noise` (0.2-0.3), raise `pad_s`, lower `stop_ratio` (0.3), try `gate="diff"` or `"fused"` |
| Skip more footage (precision / speed) | raise `k`, raise `min_duration_s`, lower `pad_s` |
| Busy indoor scenes | lower `q_noise`, or a fixed per-camera threshold |
| Small / distant people | raise `width` (480-640), lower `blur_ksize` (3) |
| Lighting false alarms (sun, clouds) | keep `mog2_illum_ncc=0.7`, add `ignore_polygons` for windows / sky |
| People eaten by the lighting filter | raise `mog2_illum_ncc` (0.8-0.9) or `None` |
| People who stand still count as active | raise `mog2_history` (e.g. 1500 = 5 min) |

---

## Stage 2: keyframes (`keyframes.py`)

**Job:** a few frames per segment that together show what happened. These are what a captioner and a search
index would see. Runs on the cached 5-samples/s fingerprints, so it costs almost nothing; only the chosen frames
are re-read at full resolution and saved as JPGs.

### 2.1 How "the scene changed" is measured (`tile_phash`, `phash_distance`)

A **perceptual hash** (pHash): shrink an image to 32×32, take the discrete cosine transform, keep the 8×8
coarsest coefficients, and set 64 bits by "above the median or not". Similar images → similar bits. We split the
frame into a **4×4 grid** (`hash_grid=4`) and hash each tile, so a person crossing one tile changes that tile's
hash clearly. Distance = **mean number of differing bits per tile** (0-64). Rough scale from earlier tests:
0.2 s apart ≈ 1 bit, quiet 5 s ≈ 2-3.5 bits, busy 5 s ≈ 6-8 bits.

`metric="hist"` switches to a grey-level histogram distance (Bhattacharyya, 0-1) that ignores *where* things are.

### 2.2 The rules (`select_keyframes`)

Inside each segment, walking forward through the samples:

| Reason | Rule |
|---|---|
| `first` | the segment's first sample |
| `change` | distance to the **last kept** keyframe ≥ `change_threshold`, and at least `min_gap_s` later |
| `maxgap` | no change, but `max_gap_s` passed since the last keyframe |
| `peak` | the sample with the highest motion score (of the chosen gate) |

Then **dedup** (`_dedup`): a candidate whose distance to the previous kept keyframe of the same segment is
< `dedup_threshold` is dropped, keeping the more important reason (peak > first > change > maxgap > heartbeat).
Then **heartbeat**: outside segments, one frame whenever `heartbeat_s` passed since the last keyframe of any kind.

| Setting | Value | What it does | Raise it → | Lower it → |
|---|---|---|---|---|
| `metric` | "phash" | change measure ("phash" or "hist") | | |
| `change_threshold` | 6.0 | mean bits per tile that count as "changed" | fewer keyframes, small actions missed | more keyframes |
| `min_gap_s` | 3.0 | minimum time between keyframes in a segment. **The main control over the count** | fewer keyframes, short actions fall between them | more keyframes |
| `max_gap_s` | 30.0 | maximum time without a keyframe in a segment | sparser in calm segments | denser |
| `dedup_threshold` | 2.5 | closer than this to the previous keyframe = duplicate | removes more | keeps near-duplicates |
| `heartbeat_s` | 300 | one idle frame per 5 min outside segments (0 = off) | | |

`naive_fps=1.0` (batch) defines the baseline "1 frame per second" used for the reduction factor.

### 2.3 What happened in the run

| | Keyframes | Per minute | Fewer than 1 frame/s |
|---|---|---|---|
| MEVA (1 h) | 302 | 5.0 | 11.9× |
| UCA (1 h) | 735 | 12.2 | 4.9× |

In a busy video almost all keyframes are `change` (G421 15:55: 46 change, 5 first, 5 peak, 1 heartbeat).

**The main finding of the run:** gated keyframes caught **no more labelled moments than the same number of
frames spread evenly** over the video.

| | Labelled items with a keyframe | Same number of frames, evenly spaced | 1 frame per second |
|---|---|---|---|
| MEVA | 48.8 % | 49.6 % | 93 % (G421 15:55) |
| UCA | 80.9 % | **93.8 %** | |

Why:
- **Budget.** ~5 keyframes per minute with `min_gap_s=3` can't catch MEVA's many short actions. "Picks up
  object" lasts ~0.5 s: only 10 % of them got a keyframe; "talks to person" (long) 86 %.
- **Change ≠ meaning.** A hand movement in a crowd barely changes a 4×4 tile hash.
- **UCA describes quiet parts too**, and gating puts no frames there by design. Evenly spaced frames do.
- Videos where gating marked little as active (MEVA G424 15:55: 17 % active, 6 % of labels hit) lose most.

**What to try:** spend frames *by* the gate instead of cutting to a fixed few, e.g. `min_gap_s=1.0` inside
segments, or "1 frame/s inside segments, heartbeat outside". On MEVA that would still be ~2× fewer frames than
1 frame/s everywhere while getting close to its 93 %. Compare every change against the evenly-spaced baseline
(it is computed automatically).

---

## Ground-truth check (`batch.run_cpu` → `motion_gating.gate_table`, `keyframe_hits`)

Run for every video with labels, for every gate and keyframe set. Files: `gt.csv` (labels + `has_keyframe`),
`gt_eval_gates.csv`, `gt_eval_keyframes.csv`.

| Metric | Definition | Meaningful on |
|---|---|---|
| frame precision | of the samples marked active, the share inside a labelled item | MEVA only, and only as a lower bound (unlabelled motion counts as "wrong") |
| frame recall | of the samples inside labelled items, the share marked active | MEVA; UCA (labels cover everything) is just "share active" |
| event recall | share of labelled items a segment covers by **more than half**, or overlaps with temporal **IoU ≥ 0.3** | both |
| keyframe hits (`gt_events_with_keyframe`) | share of labelled items with **at least one keyframe inside** | both. **The number that matters**: an item without a keyframe can never be found by a search over keyframes |
| baselines | `uniform_same_count` (same number of frames, evenly spaced), `naive_1fps` (1 frame/s) | both |

In busy MEVA clips frame precision comes out as 1.0 (G421 15:55): labels cover nearly every moment, so anything
marked active is "inside a label". Read precision with that in mind.

---

## Stage 3: events (`gpu_stages.py`, GPU)

### 3.1 Which frames are looked at (`build_windows`, `iter_window_chunks`)

| Setting | Value | What it does |
|---|---|---|
| `events.event_fps` | 4.0 | frames decoded inside segments for stages 3 and 4 |
| `events.frame_width` | 640 | frames resized to 640 px wide for the models |
| `events.chunk` | 128 | frames processed at a time (memory bound) |
| `segments.heartbeat_s` | 300 | plus one frame every 5 min outside segments |
| `events.methods` | tracks, siglip, xclip | which event methods run |

These frames are also the **candidates** stage 4 chooses from. Tracker state resets at the start of every
segment.

### 3.2 E1: detector + tracker → track events (`YOLOWorldTracker`, `track_events`) — ran

**YOLO-World** is an open-vocabulary detector (class names given as text); **ByteTrack** links boxes across
frames into tracks with IDs.

| Setting | Value | What it does |
|---|---|---|
| `detector_model` | yolov8s-worldv2.pt | small YOLO-World model |
| `detector_vocab` | person, car, truck, bus, bicycle, motorcycle, backpack, handbag, suitcase, box, package, shopping cart, dog | what to look for |
| `detector_conf` | 0.15 | minimum box confidence (low on purpose, so the tracker gets weak boxes too) |
| `detector_imgsz` | 640 | detector input size |
| `tracker` | bytetrack.yaml | Ultralytics' ByteTrack settings |

Rules on the tracks (distances in **box heights**, so they don't depend on distance from the camera):

| Event | Rule | Settings |
|---|---|---|
| `<class>_appears` (`vehicle_appears` for vehicles) | a track lasting ≥ 1 s starts | `min_track_s=1.0` |
| `person_loitering` | person track ≥ 20 s staying within 1.5 box heights of its centre | `loiter_s=20`, `loiter_radius=1.5` |
| `running` | person faster than 1.2 box heights/s for ≥ 0.75 s | `run_speed=1.2` |
| `object_left` | bag / box / suitcase / package track ≥ 10 s that moves < 0.3 box heights | `static_s=10`, `static_radius=0.3`, `carried_objects` |
| `crowd` | ≥ 5 people at once for ≥ 2 s | `crowd_n=5`, `crowd_min_s=2` |

**In the run:** MEVA clips gave 4-391 tracks each, UCA 5-102. G421 15:55: **391 tracks** in 5 min with ~25
people visible, and 328 of its 352 events are `person_appears`. Most "new people" are the same person getting a
new ID after an occlusion. `crowd` 6, `person_loitering` 2, `object_left` 2. Track events are therefore a poor
signal in crowded scenes until tracking is tuned (supervisor's plan: January).

### 3.3 E2: SigLIP frame tags + embeddings (`SigLIPEncoder`, `siglip_events`) — **FAILED TO LOAD**

Would: embed every candidate frame with `google/siglip-base-patch16-224` (768 numbers), compare it with the
event prompt bank (12 events, 1-2 prompts each) and 4 "nothing happening" prompts; an event is on where its best
prompt beats the best background prompt by more than `siglip_margin=1.0` logit after smoothing over
`siglip_smooth=3` samples. **The same embeddings are the search index.**

**In the run:** `ImportError: SiglipTokenizer requires the SentencePiece library`. No tags, no embeddings. This
also switches off three of the four stage-4 samplers and leaves the hand-off without embeddings. It is the same
failure as in the earlier café run.

### 3.4 E3: X-CLIP clip tags (`XCLIPScorer`, `xclip_events`) — ran, contributed nothing visible

8-frame clips (`xclip_frames=8`), a new clip every 1 s (`xclip_stride=4` at 4 fps), scored against the same
event bank + "nothing is happening" with a **softmax over 13 prompts**. An event needs probability >
`xclip_threshold=0.35` and above the background prompt.

**In the run:** loaded and ran (`[05:53:50] loaded xclip on cuda`). In the video inspected in `explore.ipynb`
all event keys are tracker keys. With 13 prompts sharing one softmax, 0.35 is rarely reached (also seen in the
café run: 0 events). Options: compare against the background prompt with a margin (like SigLIP), lower the
threshold, or drop X-CLIP.

### 3.5 Merging, importance, anchors (`merge_events`, `importance_curve`, `event_anchors`)

- **Merge:** same source + key + segment, less than `merge_gap_s=2.0` s apart → one event (`*_appears` events
  are never merged).
- **Support:** how many different sources (tracks / siglip / xclip) saw the same key overlapping in time.
  With SigLIP down and X-CLIP silent, every event had support 1.
- **Importance** per candidate frame (0-1): `0.5 × clip(motion_ratio / 3, 0, 1)` where motion_ratio = MOG2 score
  / its start threshold; raised to an event's weight where an event is on (track `*_appears`: 0.8 × score;
  others: score × (0.4 + 0.2 × min(support, 3))); set to 1.0 within 0.5 s of every track start; 0 for idle
  frames. In crowded scenes the track-start rule puts importance at 1 almost everywhere.
- **Anchors** (frames that must be kept by kcenter / importance): peaks of events with support ≥ 2, any track
  event, or score ≥ 0.7; and every track start.

---

## Stage 4: frame selection and hand-off (`gpu_stages.py`, run by `batch.run_gpu`)

All samplers choose from the same candidates (3.1), so they are comparable.

| Sampler | Rule | Settings | In the run |
|---|---|---|---|
| `phash` | keep a frame when its **whole-frame** 64-bit pHash differs from the last kept one by ≥ 10 bits and ≥ 1 s passed, or 30 s passed | `phash_change=10`, `min_gap_s=1.0`, `max_gap_s=30` | ✅ ran; **became the primary** |
| `novelty` | keep a frame if its SigLIP embedding is far (cosine distance > τ) from every frame kept so far in the segment; τ = max(0.10, 90th percentile of step distances) | `novelty_tau=0.10`, `novelty_adaptive_q=0.9` | ⏭️ skipped (no embeddings) |
| `kcenter` | start from anchors, repeatedly add the frame worst covered (85 % embedding distance + 15 % time distance), weighted by importance; stop at the budget or when every frame is within 0.06 | `kcenter_lambda_t=0.15`, `kcenter_radius=0.06` | ⏭️ skipped |
| `importance` | spread the budget along the importance curve (dense where important), add anchors, drop near-duplicates (similarity > 0.97) | `dedup_sim=0.97` | ⏭️ skipped |
| `v1_keyframes` | stage 2's keyframes moved to the nearest candidate frame, for comparison | — | ✅ |

Budget for `kcenter` / `importance` per segment: `round(rate_per_min × minutes × (0.5 + mean importance))`,
clipped to [`min_per_segment=2`, `max_per_segment=40`], `rate_per_min=8`.

**Metrics** (`sampling_comparison.csv`): keyframes, per minute, reduction vs 1 frame/s, coverage and redundancy
(need embeddings → NaN in this run), event recall and track recall (against the pipeline's **own** events, so
partly circular for samplers seeded with anchors). The fair comparison is `gt_eval_sampling.csv` (against labels).

**In the run (G421 15:55):** `phash` 72 frames, 60 % of labels hit; `v1_keyframes` 57 frames, 57 %. Same story as
stage 2: a few more frames, a few more hits, no better than even spacing.

**Hand-off** (`write_index_records`), for the primary sampler (`phash`, because kcenter didn't run):
- `events/index_records.jsonl`: one record per frame: camera, video, frame, time, **wall-clock time** (MEVA:
  from the file name), segment, importance, events at that moment, objects (track classes and counts),
  `siglip_tag`, `embedding_row`. 1,075 records in total (290 MEVA, 785 UCA).
- `events/index_embeddings.npy`: **empty in this run** (no SigLIP).
- No stage-4 JPGs (only the kcenter sampler exports them).

This is what the Postgres + pgvector database will ingest. Without embeddings it can't be searched yet.

## Stage 5: captions (`captions.py`, run by `batch.run_vlm`) — not run yet

Added 26 Nov 2026 after the run above; tested only with a fake captioner (`"mock": True`), so the real model's
speed and caption quality are still unknown. Method and reasons: `docs/pipeline.md` section 4b.

**What it does, per video:** decode every frame it needs once and save it at model size in `captions/frames/`,
then (1) one VLM call per selected frame (batched), (2) one VLM call per clip (≤ 6 frames in time order). Each
answer is parsed as JSON; an answer that isn't valid JSON is kept as plain text (`parse_ok=False`) so the frame
stays searchable.

| Setting (`CFG["vlm"]`, defaults in `captions.CAPTION_CONFIG`) | Default | Raise it → | Lower it → |
|---|---|---|---|
| `model` | `Qwen/Qwen2.5-VL-7B-Instruct` | (32B won't fit one L4) | `Qwen2.5-VL-3B-Instruct`: ~2× faster, ~8 GB, weaker captions |
| `levels` | `["frames", "clips"]` | — | `["clips"]` only: far fewer calls (one per ≤ 60 s instead of per frame) |
| `source` | `"auto"` (stage 4 if present, else stage 2) | — | — |
| `frame_max_pixels` | 1280×28×28 ≈ 1.0 MP | small far-away people readable; more tokens, slower | faster; small people become blobs |
| `clip_max_pixels` | 448×28×28 ≈ 0.35 MP per frame | more detail per clip frame | faster, fits more frames |
| `clip_max_frames` | 6 | sees more of a long clip; slower, more VRAM | faster; may miss short actions |
| `clip_min_frames` | 3 | more evenly spaced filler frames | trusts stage 2/4's choice more |
| `clip_max_s` | 60 s | fewer, longer clips; coarser timestamps | more clips, finer timestamps, more calls |
| `batch_size` | 4 | faster if VRAM allows | use 1-2 if CUDA runs out of memory |
| `max_new_tokens_frame` / `_clip` | 400 / 500 | fewer cut-off JSON answers in busy scenes | faster |
| `review_n` (top level of `CFG`) | 60 | tighter hallucination estimate | quicker hand check |

Changing any setting except `batch_size` and `levels` makes `settings_hash` change, and the next run captions
that video again. Same settings → finished videos are skipped and an interrupted video continues where it stopped.

**What to look at after a run:** `vlm_parse_failed` in `batch_summary.csv` (should be ~0), `vlm_s_per_frame`,
`vlm_min_per_video_hour` and `vlm_peak_vram_gb` (for the report's indexing-speed number), then open
`outputs/batch/caption_review.html` next to `caption_review.csv` and fill in the empty columns.

---

## Outputs (per video: `outputs/batch/<dataset>/<video>/`)

| File | From | Contents |
|---|---|---|
| `cache/scan_<hash>.npz` | 1 | cached decode pass (scores + fingerprints) |
| `motion_scores.csv` | 1 | every sample: time, `score_diff`, `score_mog2`, `score_fused`, brightness, `active_<gate>` |
| `segments.csv` | 1 | segments of the chosen gate |
| `keyframes.csv`, `keyframes/*.jpg` | 2 | keyframes of the chosen gate, with reason |
| `gt.csv`, `gt_eval_gates.csv`, `gt_eval_keyframes.csv` | GT | labels and scores |
| `summary_cpu.json`, `summary_gpu.json` | batch | this video's row in `batch_summary.csv` |
| `error_cpu.json`, `error_gpu.json` | batch | only if something failed (none in this run) |
| `events/events.csv`, `tracks.csv`, `tracks_raw.csv`, `xclip_windows.csv` | 3 | events, track summaries, raw boxes, X-CLIP probabilities |
| `events/candidate_frames.csv` | 3 | the 4 fps candidates with importance |
| `events/keyframes_<sampler>.csv`, `sampling_comparison.csv`, `gt_eval_sampling.csv` | 4 | sampler outputs and scores |
| `events/index_records.jsonl`, `index_embeddings.npy` | 4 | hand-off |
| `events/timeline.png` | 4 | one-page timeline of gates, events and samplers |
| `captions/frame_captions.jsonl`, `clip_captions.jsonl` | 5 | one JSON record per frame / clip: fields, `search_text`, raw answer, timing |
| `captions/captions.csv` | 5 | both levels in one table (camera, start/end, wall clock, text, `search_text`, image): what a search reads |
| `captions/frames/*.jpg` | 5 | the frames at the size the VLM saw them |
| `summary_vlm.json`, `error_vlm.json` | batch | stage 5's row / error |

Batch level (`outputs/batch/`): `batch_summary.csv`, `selection_used.csv`, `batch_config.json`,
`caption_review.csv` + `caption_review.html` (hand check of caption quality).

---

## Speed (hours of video per compute hour)

| | CPU stages 1-2 | GPU stages 3-4 |
|---|---|---|
| MEVA | 4.9 min per video-hour (12× real time) | 4.0 min per video-hour (15× real time) |
| UCA | 2.9 min per video-hour (21× real time) | 3.2 min per video-hour (19× real time) |

Stage 3 is almost all of the GPU time (stage 4 < 2 s per hour). These numbers are **without** SigLIP (which
will add time once it loads) and **without** any captioning, which will be the expensive step.

---

## Everything that was skipped, failed or needs checking

1. ❌ **SigLIP didn't load** (missing `sentencepiece`): no search embeddings, no SigLIP tags, 3 of 4 samplers
   skipped, hand-off without embeddings, coverage metrics NaN. Fix: `pip install sentencepiece protobuf` in
   `/mnt/5T_storage/clipsExtraction/tools/awsenv`, restart the kernel, run with `STAGES=["gpu"]`, `FORCE=True`.
2. ⚠️ **Primary sampler fell back** from `kcenter` to `phash`; **no stage-4 JPGs** were written.
3. ⚠️ **X-CLIP** ran but no X-CLIP events are visible; its softmax threshold is likely never met.
4. ⏭️ **Deep gate off** (by choice).
5. ⚠️ **Keyframes don't beat evenly spaced frames** at the same budget (48.8 vs 49.6 % MEVA, 80.9 vs 93.8 % UCA).
6. ⚠️ **Busy scenes:** the adaptive threshold rises to ~4.7 % of pixels in G421, so crowds count as inactive.
7. ⚠️ **Lighting filter** removes some real people near bright windows.
8. ⚠️ **Tracking:** ~390 IDs for ~25 people in 5 min; `person_appears` dominates the events.
9. 🔍 **Check:** G421 15:55, score exactly 0 for the last ~50 s with a crowd present (frozen frames?).
10. 🔍 **Environment:** the notebook ran in Python 3.13 (`awsenv`), not the `motion-gating` env. Fine, but install
    GPU packages there and note it in the README.
11. ⏭️ UCA `prefer_split="test"` had no effect (all dev videos are `val`).
12. Harmless warnings in the log: `torch.jit.load is deprecated`, `IProgress not found`, X-CLIP config
    `bos_token_id/eos_token_id` messages.

---

## Suggested experiments, cheapest first

Each is a config change + re-run. Compare in `explore.ipynb` sections 1 and 3 against the evenly-spaced baseline.

| # | Change | Expected effect | Re-runs |
|---|---|---|---|
| 1 | install `sentencepiece`, `STAGES=["gpu"]`, `FORCE=True` | embeddings, kcenter primary, SigLIP tags | GPU phase only (~7 min) |
| 2 | `gate="diff"` (or `"fused"`) | +2 to +7 points of labels hit (from 1.7) | segments + keyframes only (scan is cached) |
| 3 | keyframes `min_gap_s=1.0` | more frames, closer to the 1 frame/s ceiling; check it beats even spacing | keyframes only |
| 4 | `q_noise=0.2` or `k=2.0` | busy scenes get segments where crowds move | segments + keyframes |
| 5 | `mog2_illum_ncc=0.8` / `None` | fewer real people removed near windows | full scan (re-decode) |
| 6 | X-CLIP: margin over background instead of 0.35 | X-CLIP events appear (or decide to drop it) | GPU phase |
| 7 | `STAGES=["vlm"]` on 2-3 videos first, then all | first captions, s/frame, VRAM; then the hand check | VLM phase only |
| 8 | `CFG["vlm"]["model"]` = a 3B model, compare the review sheet | speed vs caption quality | VLM phase only |

Changing a stage-1 scan setting (1.1-1.3 tables) re-decodes every video (≈8 min for 2 h); changing thresholds,
segments, keyframes or the gate reuses the cache. Use `FORCE=True` either way, otherwise finished videos are
skipped.
