# How the REWIND ingest pipeline works

What every step of `pipeline.ipynb` does, why it is there, which settings matter, where it goes wrong, and which
paper it comes from. Written for the team and as raw material for the Practicum report's methodology and
related-work sections.

**How to use this document**
- Every section ends with **Check yourself** questions. Whoever owns a section answers them by reading the code
  and the cited paper, fixes anything here that turns out wrong, and puts their name in the table at the end.
- Numbers in brackets like [3] point to the references at the bottom. Where a step is our own (or v2's) idea and
  not from a paper, it says so. Don't cite a paper for something the paper doesn't say.
- Code references are `file.function`. Settings are given as `NAME=default`.

**What is verified and what isn't (18 Nov 2026).** The code runs end to end on a synthetic test video, with fake
models and with no GPU packages installed. It has **not** yet run with the real models on a GPU after the merge.
The only real-footage results are from before the merge (section 6).

---

## 0. The big picture

A fixed camera records hours where nothing happens. Running a captioning model or building a search index over
every frame is slow and expensive, and most of those frames are the same empty scene. The pipeline therefore
narrows the video down in four steps, each more expensive than the last but running on less of the video:

```
video (30 frames/s)
  │
  ├─ Stage 1  motion gating        every 1/5 s, cheap (CPU)      → segments: time ranges with activity
  ├─ Stage 2  keyframes            inside segments, cheap (CPU)  → a few representative frames per segment
  ├─ Stage 3  events               inside segments, GPU          → who/what is there, what happens
  └─ Stage 4  frame selection      inside segments, GPU          → frames for the captioner + search index
                                                                   (index_records.jsonl + embeddings)
```

| Stage | Code | Origin |
|---|---|---|
| 1 | `motion_gating.py` (+ `gpu_stages.scan_deep` for the deep gate) | v1 (Farid), with v2's threshold, colour MOG2 and fused vote |
| 2 | `keyframes.py` | v1 |
| 3, 4 | `gpu_stages.py` | v2 (Muzayyan, written with an LLM) |

This matches the supervisor's framing of what makes surveillance different: hours of nothing (so skipping empty
footage matters), rare and short events, and results that must point to an exact timestamp.

---

## 1. Stage 1: motion gating (`motion_gating.py`)

**Goal:** mark the time ranges where something moves. Missing an event here is fatal (later stages never see
it), while a false alarm only costs some extra compute. So stage 1 should favour **recall** over precision.

### 1.1 Sampling and preprocessing (`scan_video`, `preprocess`)

- Only every 6th frame is analysed (`sample_fps=5.0` on a 30 fps video). Neighbouring frames are nearly
  identical, so this is 6× less work for almost no loss. Every frame still has to be *decoded* (video codecs
  can't skip ahead cheaply), but only the kept ones are converted to images (`video_utils.iter_sampled_frames`).
- Frames are shrunk to 320 px wide (`width=320`) and blurred with a 5×5 Gaussian (`blur_ksize=5`). Motion
  detection needs neither detail nor full resolution, and blurring averages away sensor and compression noise.
- Frame differencing uses the greyscale image; MOG2 uses the colour image (1.3).
- The decode pass is the slow part and is cached in `outputs/<video>/cache/`, so changing thresholds later
  takes about a second.

### 1.2 Frame differencing (`MotionScorer.update`)

A pixel "moves" if its grey level changed by more than 15 (of 255) since the previous sample
(`diff_threshold=15`). Simple and fast, but it only sees things that are moving *right now*: a person who stops
disappears, and a slow-moving object barely registers. It is a baseline, not the default.

### 1.3 MOG2 background subtraction (`MotionScorer`, OpenCV `createBackgroundSubtractorMOG2`)

For every pixel, MOG2 keeps a small **mixture of Gaussians** (bell curves) describing the values that pixel
usually has, and updates it with every new frame. A value that fits none of the curves is foreground. The number
of curves per pixel adapts automatically. This is Zivkovic's method [1, 2], which is what OpenCV implements.

- `mog2_history=500` samples = 100 s of memory at 5 samples/s. Slow changes (the sun moving) are absorbed into
  the background, but so is **a person who stands still for about that long**. "Inactive" therefore means "no
  significant motion", not "no people".
- `mog2_var_threshold=16`: how far (in squared standard deviations) a value may be from a curve and still fit.
  Lower = more sensitive.
- **Shadow detection.** OpenCV marks a pixel as shadow (value 127 instead of 255) when it is a darker version of
  the background: brightness between about 50% and 100% of the background, with the same colour. The idea comes
  from Prati et al. [3], which the OpenCV documentation cites. We discard shadow pixels.
- **Why MOG2 runs on colour (`mog2_color=True`).** On a greyscale image there is no colour to compare, so *any*
  pixel 50-100% as bright as the background passes as a shadow, including a person in mid-dark clothes on a light
  floor. v1 ran on greyscale; on a synthetic test it missed a walking person completely. v2 ran on colour, and so
  does the merged code. People in dark *grey or black* clothes on a grey floor can still be taken for shadows in
  colour too (same hue, darker), so this is improved, not solved.

### 1.4 Clean-up (`MotionScorer._clean`)

A morphological **opening** (erode, then dilate, 3×3 ellipse, `morph_kernel=3`) removes isolated noise pixels
but keeps real blobs [4]. Regions in `ignore_polygons` (e.g. the sky) are masked out.

### 1.5 Lighting filter (`remove_illumination_blobs`, MOG2 only)

When a cloud passes, a patch of grass gets darker but keeps its texture. When a person walks in, the texture is
replaced. For each foreground blob, the filter compares the current pixels with MOG2's background image using
**normalised cross-correlation** (NCC: subtract each patch's mean, then correlate; 1 = same pattern). Blobs with
NCC above `mog2_illum_ncc=0.7` are treated as lighting and removed.

This is our own heuristic built on standard NCC, not taken from a paper. On the MEVA test video it reduced a
sunlit-lawn false alarm from 9.3% to 0.3% of pixels, while people kept about 90-100% of their foreground.
A textureless region (a flat wall) has no pattern to correlate, so the filter can't judge it either way.

### 1.6 Score and adaptive threshold (`robust_threshold`, `start_stop_thresholds`)

**Score** = share of (non-ignored) pixels marked as moving, per sample.

**Threshold** = median + k × 1.4826 × MAD, with `k=3`. MAD (median absolute deviation) measures spread like the
standard deviation does, but a few large spikes, which are exactly the motion we're looking for, barely affect
it. 1.4826 × MAD equals the standard deviation for normally distributed data, so k=3 means "three standard
deviations above normal". Using median and MAD instead of mean and standard deviation for outlier detection is
the recommendation of Leys et al. [5], who suggest 2.5-3 as the cut-off. The threshold never goes below
`floor=0.001` (0.1% of pixels).

**Quieter-half rule (`q_noise=0.5`).** The median and MAD are computed only over the quietest 50% of samples.
Without this, a busy camera breaks the method: if something moves more than half of the time, the median *is*
motion and the threshold rises above real activity. On the café video, v1 (all samples) marked people walking
across the room as inactive. This rule is v2's heuristic, not from a paper. Its assumption: at least half of the
time is "normal background". `q_noise=1.0` restores the old behaviour.

### 1.7 Segments (`find_segments`)

1. Smooth the score with a 1 s rolling median (`smooth_s=1.0`).
2. **Hysteresis:** a segment starts when the score rises above the threshold and ends only when it drops below
   half of it (`stop_ratio=0.5`), so it doesn't flicker on and off around one value. The two-threshold idea is
   the same as in Canny's edge detector [6].
3. Merge segments less than 3 s apart (`merge_gap_s`), drop segments under 1 s (`min_duration_s`), and add 2 s
   before and after (`pad_s`), so an event's start and end aren't cut off.

### 1.8 Deep gate: DINOv2 patch features (`gpu_stages.scan_deep`, `DeepFeatureGate`)

An optional third motion score, from v2. DINOv2 [7] is a vision transformer trained without labels to produce
general-purpose image features; it cuts an image into 14×14-pixel patches and gives each patch a feature vector
(`facebook/dinov2-small`, input height 224).

v2's gate (its own design, not from the DINOv2 paper): every patch keeps a slowly updated "background" vector. A
patch counts as changed when its cosine distance to that background is both unusually large for that patch
(z-score > 3, from a running mean and variance) and larger than 0.05. Score = share of changed patches. Runs at
2 frames/s and is held constant in between.

The hope is robustness to lighting and compression, since features describe content rather than raw brightness.

**Weak spots seen on the café video:** the first frames become the background, so people present at the start
count as "change" until the slow update rate (`deep_alpha_fg=0.008`) absorbs them, about a minute. Its agreement
with the other gates was low (IoU ≈ 0.33), and its recall against the ground truth was the lowest of the gates.
It also needs a GPU for the whole video, which defeats some of the purpose of a cheap first stage.

### 1.9 Fused gate (`add_fused`)

Each gate's smoothed score is divided by its own start threshold (1.0 = exactly at the threshold), and the fused
score is the **mean of the two highest ratios**: a soft "two gates agree" vote. v2's heuristic. With only diff and
MOG2, it is simply their average ratio.

**Default gate: MOG2.** On the café ground truth, MOG2 and fused were nearly tied (section 6), and fused needs the
deep gate's GPU pass. Change it with `"gate"` in the config cell of `pipeline.ipynb`.

### Check yourself (stage 1)
1. Why can stage 1 afford false alarms but not misses? What does each cost downstream?
2. In [1], how does MOG2 decide how many Gaussians a pixel needs? Which OpenCV parameter controls how fast it forgets?
3. Why does greyscale MOG2 call a person in a dark-red shirt a "shadow", and why doesn't colour MOG2? Which clothes can still fool it?
4. Compute the threshold by hand for the scores [0, 0, 0.001, 0.002, 0.05, 0.08] with k=3, first with `q_noise=1.0`, then with 0.5.
5. What happens to a person who sits still for 3 minutes? Is that a problem for our use case?
6. Why is the deep gate's first minute unreliable? How could that be fixed?

---

## 2. Stage 2: keyframes (`keyframes.py`)

**Goal:** a few frames per segment that together show what happened, for the captioner and the index.

For each segment (`select_keyframes`):
- **first**: the segment's first sample.
- **change**: walking forward, a new keyframe whenever the image differs enough from the last keyframe, and at
  least `min_gap_s=3` s later.
- **maxgap**: a forced keyframe if `max_gap_s=30` s pass without one, so long segments stay covered.
- **peak**: the sample with the highest motion score.
- **dedup** (`_dedup`): candidates closer than `dedup_threshold=2.5` to an already kept one are dropped, keeping
  the more important reason (peak > first > change > maxgap).

Outside segments, one **heartbeat** frame every `heartbeat_s=300` s, so slow changes (a car parked for an hour)
still reach the index.

**How "differs enough" is measured.** A **perceptual hash** (pHash) turns an image into a 64-bit fingerprint of
its coarse structure: shrink it, take the discrete cosine transform, and set each bit by whether a
low-frequency coefficient is above the median. Similar images get similar bits; distance = number of differing
bits (Hamming distance). Zauner [8] benchmarks this DCT-based hash among others. Our addition: the frame is split
into a 4×4 grid with one hash per tile (`tile_phash`), and the distance is the mean over tiles
(`change_threshold=6` bits). A whole-frame hash barely changes when a small person walks across a large scene;
a tile hash does.

`min_gap_s` is the main control over how many keyframes come out.

**Weak spots:** keyframe timing is tied to *visual* change, not to *meaning*: a person turning around and a
person pulling out a weapon can produce similar hash changes. That is what stage 4's SigLIP-based samplers try
to address.

### Check yourself (stage 2)
1. Why does a 4×4 tile hash notice a small moving person when a single 64-bit hash doesn't?
2. What would happen to the keyframe count if `min_gap_s` went from 3 to 10 s?
3. Why does the heartbeat exist, given that nothing moves outside segments?

---

## 3. Stage 3: events (`gpu_stages.py`, GPU)

Runs only inside segments, on frames decoded at 4 per second (`event_fps=4`) and resized to 640 px. Three
independent methods, each skipped with a warning if its model can't load (`Models`).

### 3.1 Detector + tracker → track events (`track_events`)

- **YOLO-World** [9] (`yolov8s-worldv2.pt`) is an *open-vocabulary* detector: instead of a fixed class list, it
  takes the class names as text and matches them against image regions, so we can ask for "box" or "shopping
  cart" without retraining. Our list: person, car, truck, bus, bicycle, motorcycle, backpack, handbag, suitcase,
  box, package, shopping cart, dog. Confidence threshold 0.15 (low, so the tracker gets weak boxes too).
- **ByteTrack** [10] links detections across frames into tracks with IDs. Its key idea: also use low-confidence
  boxes in a second matching round instead of discarding them, which keeps partly hidden people tracked.
- **Rules** on top of the tracks (v2's own heuristics; distances are measured in box heights, so they don't
  depend on how far the person is from the camera):

  | Event | Rule (defaults) |
  |---|---|
  | `<class>_appears` | a track at least 1 s long starts |
  | `person_loitering` | a person track ≥ 20 s that stays within 1.5 box heights of its centre |
  | `running` | a person moving faster than 1.2 box heights per second for ≥ 0.75 s |
  | `object_left` | a bag/box/suitcase track ≥ 10 s that doesn't move (< 0.3 box heights) |
  | `crowd` | ≥ 5 people at once for ≥ 2 s |

**Weak spots (café video, 5 min):** 205 tracks with only around 15 people visible at a time, so track IDs switch often in crowds,
and most `person_appears` events are the same person getting a new ID. In a busy lobby, `crowd` is on almost all
the time and people waiting trigger `loitering`. The thresholds were written for outdoor CCTV and need tuning per
scene; the supervisor's plan puts proper detection + tracking work in January.

### 3.2 SigLIP frame tags (`siglip_events`)

**SigLIP** [11] (`google/siglip-base-patch16-224`) is an image-text model in the CLIP family [12]: it maps images
and sentences into the same vector space, so an image's similarity to "a person carrying a box" can be measured
without training a classifier. SigLIP replaces CLIP's softmax training loss with a per-pair sigmoid loss, which
means each image-text score stands on its own rather than being relative to the other texts in a batch.

v2 compares every frame with a prompt bank (12 events, e.g. walking, carrying, running, fighting, falling, crowd,
vehicle parking, object left, delivery, door, cycling) and with "nothing happening" prompts ("an empty room",
"a quiet scene with no people"...). An event is detected where its best prompt beats the best background prompt
by more than 1.0 logit (`siglip_margin`) after light smoothing.

Two jobs: the tags become events, and **the image embeddings (768 numbers per frame) are what the search index
stores**. The tag list only labels things; the search later uses the raw embeddings, so anything describable
stays findable even if it's not in the list.

**Status:** in the only real run, SigLIP failed to load because its tokenizer needs the `sentencepiece` package.
So it has never produced results on real footage yet.

### 3.3 X-CLIP clip tags (`xclip_events`)

**X-CLIP** [13] (`microsoft/xclip-base-patch32`) extends CLIP to short video clips (8 frames), so it can see
motion over time (falling, running) that a single frame can't. Clips are taken every second inside segments and
compared with the same prompt bank plus "nothing is happening"; probabilities come from a softmax over the 13
prompts. An event needs probability > 0.35 and above the background prompt.

**Weak spots:** this checkpoint was trained (fully supervised) on Kinetics-400 [14], 400 fixed action classes, so
its "zero-shot" scores on our own prompts are less reliable than the name suggests. And with a softmax over 13
prompts, probability spreads thin and rarely passes 0.35. In the café run it produced **0 events**, and it gave
"vehicle parking" 0.29 in an indoor scene. Treat it as experimental.

### 3.4 Merging and importance (`merge_events`, `importance_curve`, `event_anchors`)

- Events with the same method, label and segment that are less than 2 s apart are merged.
- **Support** = how many different methods saw the same label overlapping in time (1-3). Agreement between
  independent methods is the main evidence that an event is real.
- **Importance curve** (0-1 per candidate frame): half from motion (the gate's score relative to its threshold),
  raised by events (more for higher support), and set to 1 around the start of every track. Idle frames get 0.
- **Anchors**: frames that must be represented: peaks of events with support ≥ 2, any track event, or score ≥ 0.7,
  and track starts.

### Check yourself (stage 3)
1. What makes YOLO-World "open-vocabulary", and what does that let us do that a COCO-trained YOLO can't [9]?
2. What problem does ByteTrack's second matching round solve [10]? Why do IDs still switch in a crowd?
3. Why does SigLIP's sigmoid loss make a single image-text score more meaningful on its own than CLIP's softmax [11]?
4. Why is the X-CLIP checkpoint's zero-shot behaviour limited [13, 14]? What would a fairer test be?
5. Which café findings suggest the track rules need per-scene tuning?

---

## 4. Stage 4: which frames go to the captioner and the index (`gpu_stages.py`)

All samplers choose from the **same candidates** (stage 3's 4 frames/s inside segments, plus idle heartbeats),
so the comparison is fair. Budget for the embedding-based samplers (`_budget`): about 8 frames per active minute
(`rate_per_min=8`), scaled by the segment's mean importance, between 2 and 40 per segment.

| Sampler | How it chooses | Origin |
|---|---|---|
| `phash` | like stage 2's "change", with one global 64-bit hash (≥ 10 differing bits, ≥ 1 s apart) | v1 idea, v2 code |
| `novelty` | keep a frame if its SigLIP embedding is far (cosine distance > τ) from **every** frame kept so far in the segment; τ adapts per segment | v2 heuristic |
| `kcenter` (primary) | greedy **k-center**: start from the anchors, then repeatedly add the frame worst covered by the selection, weighted by importance; distance mixes SigLIP distance (85%) and time (15%); stops early once every frame is close to a kept one | k-center greedy [15], as used for core-set selection [16]; weighting is v2's |
| `importance` | spread the budget along the importance curve (inverse-CDF sampling: dense where important), add anchors, drop near-duplicates (similarity > 0.97) | v2 heuristic on a standard technique |
| `v1_keyframes` | stage 2's keyframes moved to the nearest candidate, so v1 is scored the same way | added in the merge |

The greedy k-center algorithm picks points so that the largest distance from any point to its nearest chosen
point is small; Gonzalez [15] showed this simple greedy rule is within a factor of 2 of the best possible.
Sener & Savarese [16] used the same idea to pick a small, representative subset of a dataset. Here it means:
choose frames so that no moment of the segment looks very different from every chosen frame.

**Metrics** (`evaluate_sampling`): coverage (mean and worst 10% of each candidate's similarity to its closest kept
frame, in SigLIP space), redundancy (how similar kept frames are to each other), event recall and track recall
(share of stage 3's events / tracks with at least one kept frame inside them).

**Two cautions for the report:**
- Event and track recall are measured against **events the same pipeline detected**, not against human labels.
  `kcenter` and `importance` are seeded with those events' anchors, so part of their event recall holds by
  construction. Ground-truth recall (section 5) is the fair comparison.
- Coverage and redundancy need SigLIP embeddings. Without SigLIP, only `phash` and `v1_keyframes` run.

**Hand-off** (`write_index_records`): for the primary sampler, `events/index_records.jsonl` (one JSON object per
frame: camera, video, frame number, time, wall-clock time if known, segment, importance, events, objects, SigLIP
tag, `embedding_row`) and `events/index_embeddings.npy` (the 768-number SigLIP vectors, row = `embedding_row`).
This is what the Postgres + pgvector database ingests.

### Check yourself (stage 4)
1. Explain k-center greedy in two sentences. What does the factor-2 guarantee in [15] mean?
2. Why does "far from every kept frame" (novelty) avoid repeats that "far from the last kept frame" doesn't?
3. Why is `kcenter`'s event recall partly circular, and what should we report instead?

---

## 5. Evaluation

### 5.1 What the code measures now (`motion_gating.gate_table`, `keyframe_hits`, `gate_agreement`)

For every video with labels (from the dataset annotations, section 5.2a; any `start_s,end_s[,label]` table works),
the batch saves, and `explore.ipynb` shows:
- per gate: **frame precision / recall / F1** (share of sampled moments correctly marked active), **event recall**
  (share of labelled events that a segment covers by more than half, or with IoU ≥ 0.3), and % of video active;
- per keyframe set (every gate's keyframes, every sampler): the share of labelled events with **at least one
  keyframe inside**. This is the number that matters downstream: an event without a keyframe is never captioned
  or indexed;
- without ground truth: how much the gates agree (IoU of their active time).

### 5.2 Data

| Dataset | What it gives us | Ref |
|---|---|---|
| **UCA** (UCF-Crime Annotation) | the supervisor's main benchmark: 23,542 sentences with start/end times over 110.7 h of UCF-Crime surveillance video. Text queries with timestamps, exactly what Recall@K needs | [17] |
| UCF-Crime | the underlying videos: real surveillance footage of 13 anomaly types | [18] |
| MEVA | multi-camera, untrimmed footage of an access-controlled site with scripted and spontaneous activity (our 30-min test video). No text queries, so we write our own | [19] |

UCA's timestamped sentences can also serve as ground truth for stage 1 and 2 ("did every annotated event get a
segment and a keyframe?"), which replaces hand-made files of unknown origin.

### 5.2a Ground truth in the batch run (`datasets.py`, `batch.py`)

The batch (`pipeline.ipynb` / `batch.py`) runs on ~1 h of MEVA and ~1 h of UCA and scores every video that has labels:

- **MEVA** `*.activities.yml` (KPF format): one activity per line with a frame range `tsr0: [start, end]`; seconds =
  frame / fps. Only ~37 activity types are labelled (carrying, opening a trunk, talking...), so unlabelled motion is
  not an error. Useful as "these moments must not be missed" (event recall, keyframe hits), not for precision.
  Only part of MEVA is annotated; `datasets.py` matches annotation files to videos by name (without the `.r13`
  release suffix, ±5 s on the start time).
- **UCA** sentences cover each video back to back, quiet parts included, so frame precision/recall against them
  mean little. The useful number is the share of sentences whose time span contains at least one keyframe: a span
  without one can never be found by a search over keyframes.
- **Baselines** in every `gt_eval_keyframes.csv`: the same number of frames spread **uniformly** over the video, and
  1 frame/s. Gating only earns its place if it beats uniform sampling at the same budget.

Known caveats: each 5-min MEVA clip is processed on its own, so MOG2 re-learns the background at the start of every
clip (`mog2_history=500` samples = 100 s); joining clips per camera would avoid that. UCF-Crime is partly not CCTV
(news/YouTube compilations with cuts and replays); a cut looks like whole-frame motion, so those videos come out
mostly "active".

### 5.3 What the supervisor asked for (not built yet)

Retrieval Recall@1/5/10 at temporal IoU 0.3 and 0.5 on UCA test queries, against CLIP-only and SigLIP-only
baselines; 50-100 of our own queries on MEVA and our footage; query latency and hours of video indexed per
GPU-hour; how often the captioner mentions things that aren't there (checked by hand). This needs the database
and search, which are outside this repo.

---

## 6. Results so far (before the merge; treat as indicative)

**Café video (5 min, busy indoor lobby).**

| | v1 (MOG2, threshold from all samples, greyscale) | v2 (fused gate, quieter-half threshold, colour) |
|---|---|---|
| MOG2 start threshold | 1.68% of pixels | 1.05% |
| share of video active | 27% | 67% |
| keyframes | 37 | 88 (pHash sampler only) |
| keyframes per active minute | ≈ 26 | ≈ 26 |

- Both pick about the same number of keyframes per minute of *activity*; v2 has more keyframes only because its
  gate marks more of the video as active.
- v1's "inactive" sample frames showed people walking (missed activity); v2's showed fewer such cases.
- Against a ground-truth file `gt_motion.csv` (**origin unknown, needs confirming**): MOG2 frame F1 0.507 and
  event recall 0.744; fused 0.500 and 0.785; all gates had frame precision ≈ 0.40; the deep gate had the lowest
  recall. Only **41%** of ground-truth events got a pHash keyframe, which is the most worrying number.
- Stage 3: SigLIP failed to load (missing `sentencepiece`), so 3 of 4 samplers were skipped; X-CLIP produced
  0 events; track events were dominated by `person_appears` (ID switches) and `crowd`.

**MEVA video (30 min, mostly idle outdoor).** v1 settings: 109 keyframes instead of 1,801 at 1 frame/s (16.5×
fewer). Not re-run with the merged defaults yet.

**Interpretation.** The café is the hardest case for gating (constant activity, little to skip); MEVA is the
typical one ("hours of nothing"). Both should stay in the development set, and the report should show both.

---

## 7. Open questions

1. Where did `gt_motion.csv` come from (hand-labelled, or generated)? Until we know, the café ground-truth
   numbers can't go in the report.
2. Re-run both videos with the merged defaults and real models on the GPU: new MEVA baseline, SigLIP working,
   all four samplers compared on ground truth.
3. Is the deep gate worth a full-video GPU pass? Current evidence says no.
4. Should stage 1 favour recall even more (lower `k`), given that misses are fatal and later stages filter?
5. Track rules: tune per scene, or replace with SigLIP/VLM-based checks?
6. Replace X-CLIP's softmax threshold with a margin over the background prompt (like SigLIP), or drop X-CLIP?
7. ~~Use UCA annotations as ground truth for stages 1-2~~: done in the batch run (section 5.2a), with MEVA activities too.

---

## Section owners

| Section | Checked by | Date | Notes |
|---|---|---|---|
| 1. Motion gating (1.1-1.7, 1.9) | | | |
| 1.8 Deep gate | | | |
| 2. Keyframes | | | |
| 3. Events | | | |
| 4. Frame selection + hand-off | | | |
| 5-6. Evaluation and results | | | |

---

## References

1. Z. Zivkovic, "Improved adaptive Gaussian mixture model for background subtraction," *ICPR*, 2004. [doi:10.1109/ICPR.2004.1333992](https://doi.org/10.1109/ICPR.2004.1333992)
2. Z. Zivkovic and F. van der Heijden, "Efficient adaptive density estimation per image pixel for the task of background subtraction," *Pattern Recognition Letters*, 2006. [doi:10.1016/j.patrec.2005.11.005](https://doi.org/10.1016/j.patrec.2005.11.005)
3. A. Prati, I. Mikic, M. Trivedi, R. Cucchiara, "Detecting moving shadows: algorithms and evaluation," *IEEE TPAMI*, 2003. [doi:10.1109/TPAMI.2003.1206509](https://doi.org/10.1109/TPAMI.2003.1206509). Cited for shadow detection in the [OpenCV MOG2 documentation](https://docs.opencv.org/5.0/main_modules/classcv_1_1BackgroundSubtractorMOG2.html).
4. OpenCV, "Morphological Transformations" (tutorial). [docs.opencv.org](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
5. C. Leys, C. Ley, O. Klein, P. Bernard, L. Licata, "Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median," *Journal of Experimental Social Psychology*, 2013. [doi:10.1016/j.jesp.2013.03.013](https://doi.org/10.1016/j.jesp.2013.03.013)
6. J. Canny, "A computational approach to edge detection," *IEEE TPAMI*, 1986. [doi:10.1109/TPAMI.1986.4767851](https://doi.org/10.1109/TPAMI.1986.4767851)
7. M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," 2023. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
8. C. Zauner, "Implementation and Benchmarking of Perceptual Image Hash Functions," Master's thesis, Upper Austria University of Applied Sciences, 2010. [phash.org PDF](https://phash.org/docs/pubs/thesis_zauner.pdf)
9. T. Cheng et al., "YOLO-World: Real-Time Open-Vocabulary Object Detection," *CVPR*, 2024. [arXiv:2401.17270](https://arxiv.org/abs/2401.17270). Model used through [Ultralytics](https://docs.ultralytics.com/models/yolo-world/).
10. Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," *ECCV*, 2022. [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
11. X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer, "Sigmoid Loss for Language Image Pre-Training," *ICCV*, 2023. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
12. A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), *ICML*, 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
13. B. Ni et al., "Expanding Language-Image Pretrained Models for General Video Recognition" (X-CLIP), *ECCV*, 2022. [arXiv:2208.02816](https://arxiv.org/abs/2208.02816). Checkpoint: [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) (trained on Kinetics-400, 8 frames, 224×224).
14. W. Kay et al., "The Kinetics Human Action Video Dataset," 2017. [arXiv:1705.06950](https://arxiv.org/abs/1705.06950)
15. T. F. Gonzalez, "Clustering to minimize the maximum intercluster distance," *Theoretical Computer Science*, 1985. [doi:10.1016/0304-3975(85)90224-5](https://doi.org/10.1016/0304-3975(85)90224-5)
16. O. Sener and S. Savarese, "Active Learning for Convolutional Neural Networks: A Core-Set Approach," *ICLR*, 2018. [arXiv:1708.00489](https://arxiv.org/abs/1708.00489)
17. T. Yuan et al., "Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges" (UCA), *CVPR*, 2024. [arXiv:2309.13925](https://arxiv.org/abs/2309.13925)
18. W. Sultani, C. Chen, M. Shah, "Real-world Anomaly Detection in Surveillance Videos" (UCF-Crime), *CVPR*, 2018. [arXiv:1801.04264](https://arxiv.org/abs/1801.04264)
19. K. Corona, K. Osterdahl, R. Collins, A. Hoogs, "MEVA: A Large-Scale Multiview, Multimodal Video Dataset for Activity Detection," *WACV*, 2021. [arXiv:2012.00914](https://arxiv.org/abs/2012.00914)
