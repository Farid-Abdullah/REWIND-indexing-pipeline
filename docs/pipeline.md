# How the REWIND ingest pipeline works

What every stage of the pipeline does, why it is there, where it goes wrong on real footage, and which paper it
comes from. Written for the team and as raw material for the Practicum report's methodology, results and
related-work sections.

- **Every setting, its default and what raising or lowering it does:** `docs/explanation.md`. This file explains
  the methods; that one is the tuning guide.
- Every section ends with **Check yourself** questions. Whoever owns a section answers them from the code and the
  cited paper, fixes anything here that turns out wrong, and signs the table at the end.
- Numbers in brackets like [3] point to the references. Where a step is our own (or v2's) idea and not from a
  paper, it says so. Don't cite a paper for something it doesn't say.
- Code references are `file.function`; settings are `NAME=default`.

**Status (19 Nov 2026).** First real run of the merged pipeline on the team VM (one L4): 38 videos, 2.00 h
(1 h MEVA, 1 h UCA), all stages, no video failed, 15.3 min wall time. **SigLIP failed to load** (missing
`sentencepiece`), so there are no search embeddings yet and three of the four stage-4 samplers were skipped
(section 4). Results: section 6.

---

## 0. The big picture

A fixed camera records long stretches where little happens. Captioning or indexing every frame is slow and
expensive, and most frames repeat the same scene. The pipeline narrows the video down in four stages, each more
expensive than the last but running on less of the video:

```
video (25-30 frames/s)
  │
  ├─ Stage 1  motion gating        5 samples/s, cheap (CPU)        → segments: time ranges with activity
  ├─ Stage 2  keyframes            inside segments, cheap (CPU)    → a few representative frames per segment
  ├─ GT check                      CPU                             → did segments / keyframes catch the labels?
  ├─ Stage 3  events               inside segments, 4 frames/s, GPU→ who/what is there, what happens
  └─ Stage 4  frame selection      same candidates, GPU            → frames for the captioner + search index
                                                                     (index_records.jsonl + embeddings)
```

| Part | Code | Origin |
|---|---|---|
| Dataset discovery, selection, labels | `datasets.py` | added for the batch run |
| Batch loop (two phases, resume, per-video errors) | `batch.py`, `pipeline.ipynb` | added for the batch run |
| Stage 1 | `motion_gating.py` (+ `gpu_stages.scan_deep` for the deep gate) | v1 (Farid), with v2's threshold, colour MOG2 and fused vote |
| Stage 2 | `keyframes.py` | v1 |
| Stages 3, 4 | `gpu_stages.py` | v2 (Muzayyan, written with an LLM) |
| Results views | `explore.ipynb` | added for the batch run |

The batch runs stages 1-2 (and the ground-truth check) for **every video first**, then stages 3-4 for every
video, so a GPU problem never costs the CPU results.

This follows the supervisor's framing of what makes surveillance different: long stretches with little happening
(so skipping footage matters), rare and short events, many cameras with real clock times, and results that must
point to an exact timestamp.

---

## 1. Stage 1: motion gating (`motion_gating.py`)

**Goal:** mark the time ranges where something moves. A miss here is fatal (later stages never see that moment),
while a false alarm only costs compute. So stage 1 should favour **recall**.

### 1.1 Sampling and preprocessing (`scan_video`, `preprocess`)

Every 6th frame is analysed (`sample_fps=5`), shrunk to 320 px wide and blurred (5×5 Gaussian). Neighbouring
frames are nearly identical, so 5 samples/s loses little; motion detection needs neither detail nor full
resolution; blurring averages away sensor and compression noise. Every frame still has to be decoded (codecs
can't skip ahead), but only samples are converted to images (`video_utils.iter_sampled_frames`). The decode pass
is cached, so threshold changes later cost about a second.

**Run:** 4.5 min of decoding per hour of MEVA (≈13× real time), 2.8 min per hour of UCA.

### 1.2 Frame differencing (`MotionScorer.update`)

A pixel "moves" if its grey level changed by more than 15 (of 255) since the previous sample. No memory: it reacts
instantly and never absorbs anyone, but only sees what moves *right now* (a person who stops disappears).

### 1.3 MOG2 background subtraction (`MotionScorer`, OpenCV `createBackgroundSubtractorMOG2`)

For every pixel, MOG2 keeps a small **mixture of Gaussians** describing the values that pixel usually has, updated
with every frame; a value that fits none of them is foreground, and the number of Gaussians per pixel adapts
automatically. This is Zivkovic's method [1, 2], as implemented in OpenCV.

- `mog2_history=500` samples = 100 s of memory. Slow changes are absorbed, and so is a person who stands still for
  about that long. "Inactive" means "no significant motion", not "no people".
- **Shadows.** OpenCV marks a pixel as shadow when it is a darker version of the background with the same colour,
  the idea of Prati et al. [3] cited by the OpenCV documentation. Shadow pixels are discarded.
- **Colour (`mog2_color=True`).** On greyscale there is no colour to compare, so any pixel 50-100 % as bright as
  the background passes as a shadow, including people in mid-dark clothes. v1 ran on greyscale and missed a
  walking person on a synthetic test; v2 and the merged code run on colour. Dark grey or black clothes on a grey
  floor can still pass as shadow.
- **MOG2 restarts per file.** Each 5-min MEVA clip is processed on its own, so the background is re-learned at the
  start of every clip and the first stretch of each clip is less reliable.

### 1.4 Clean-up (`MotionScorer._clean`)

A morphological **opening** (erode, then dilate, 3×3) removes isolated noise pixels and keeps real blobs [4].
Regions in `ignore_polygons` are masked out.

### 1.5 Lighting filter (`remove_illumination_blobs`, MOG2 only)

When a cloud passes, a patch of grass gets darker but keeps its texture; when a person walks in, the texture is
replaced. For each foreground blob, the filter compares the current pixels with MOG2's background image by
**normalised cross-correlation** (NCC); above `mog2_illum_ncc=0.7` the blob is treated as lighting and removed.
Our own heuristic on standard NCC, not from a paper. On the earlier 30-min MEVA video it cut a sunlit-lawn false
alarm from 9.3 % to 0.3 % of pixels.

**Run:** on the crowded indoor camera G421 some blobs on real people near the bright window were removed
(shown yellow in `explore.ipynb`). Flat, low-texture regions correlate easily with the background, so the
filter can remove real people. On synthetic videos with flat-coloured moving blocks it removed them entirely.

### 1.6 Score and adaptive threshold (`robust_threshold`, `start_stop_thresholds`)

**Score** = share of (non-ignored) pixels marked as moving, per sample.

**Threshold** = median + k × 1.4826 × MAD (`k=3`), never below `floor=0.001`. MAD (median absolute deviation) is a
spread measure that the spikes we are looking for barely affect; ×1.4826 makes it equal to the standard deviation
for normal data, so k=3 means "three standard deviations above normal". Median/MAD instead of mean/standard
deviation for outliers is the recommendation of Leys et al. [5], with 2.5-3 as the cut-off.

**Quieter-half rule (`q_noise=0.5`):** median and MAD come only from the quietest 50 % of samples, so a camera
that is busy *some* of the time doesn't push the threshold above real activity. v2's heuristic, not from a paper.
Its assumption, at least half the time is "background", **fails in constantly busy scenes**: on G421 15:55
(~25 people the whole clip) the start threshold came out at 4.67 % of pixels, and stretches where dozens of
people walk (≈3 %) were marked inactive.

### 1.7 Segments (`find_segments`)

1. Smooth the score with a 1 s rolling median (`smooth_s=1.0`).
2. **Hysteresis:** start above the threshold, stop only below half of it (`stop_ratio=0.5`), so a score hovering
   near one value doesn't flicker. The two-threshold idea is the same as in Canny's edge detector [6].
3. Merge segments ≤ 3 s apart, drop segments < 1 s, add 2 s before and after.

### 1.8 Deep gate: DINOv2 patch features (`gpu_stages.scan_deep`, `DeepFeatureGate`) — off

An optional third score, from v2. DINOv2 [7] is a vision transformer trained without labels whose patch features
describe image content. v2's gate (its own design): each 14×14-pixel patch keeps a slowly updated background
feature; a patch counts as changed when its cosine distance to it is unusually large for that patch (z > 3) and
above 0.05. Needs a GPU pass over the whole video. On the earlier café video it had the lowest recall and an
unreliable first minute (the first frames become its background). **Not run** in the batch (`deep_gate=False`).

### 1.9 Fused gate (`add_fused`) and the gate in use

Each gate's smoothed score divided by its own start threshold, then the mean of the two highest ratios: a soft
"two gates agree" vote (v2's heuristic). With only diff and MOG2 it is their average ratio.

**Gate in use: MOG2** (`gate="mog2"`). **In the run, frame differencing did better on both datasets** (section 6),
the opposite of what the single café video suggested, so the default is up for change.

### Check yourself (stage 1)
1. Why can stage 1 afford false alarms but not misses? What does each cost downstream?
2. In [1], how does MOG2 decide how many Gaussians a pixel needs? Which OpenCV parameter controls how fast it forgets?
3. Why does greyscale MOG2 call a person in a dark-red shirt a "shadow", and why doesn't colour MOG2? Which clothes can still fool it?
4. Compute the threshold by hand for the scores [0, 0, 0.001, 0.002, 0.05, 0.08] with k=3, first with `q_noise=1.0`, then with 0.5.
5. Why did G421 get a 4.67 % threshold, and what would `q_noise=0.2` change?
6. Why might frame differencing beat MOG2 on these videos (think: 100 s memory, busy scenes, the lighting filter)?

---

## 2. Stage 2: keyframes (`keyframes.py`)

**Goal:** a few frames per segment that together show what happened, for the captioner and the index.

For each segment (`select_keyframes`): the **first** sample; a **change** keyframe whenever the image differs
enough from the last keyframe (and ≥ `min_gap_s=3` s later); a forced **maxgap** keyframe after 30 s without one;
the **peak** motion sample; then near-duplicates are dropped (`dedup_threshold=2.5`, keeping the more important
reason). Outside segments, one **heartbeat** frame per 5 min.

**How "differs enough" is measured.** A **perceptual hash** (pHash) turns an image into a 64-bit fingerprint of its
coarse structure (shrink, discrete cosine transform, one bit per low-frequency coefficient: above the median or
not); distance = number of differing bits. Zauner [8] benchmarks this DCT hash among others. Our addition: one
hash per tile of a 4×4 grid (`tile_phash`), distance = mean over tiles (`change_threshold=6` bits), so a small
person crossing a wide scene changes one tile clearly.

**Run:** 302 keyframes for 1 h of MEVA (5 per minute, 12× fewer than 1 frame/s), 735 for 1 h of UCA (12 per minute,
5× fewer). In busy videos almost all are `change` keyframes. **They caught no more labelled moments than the same
number of evenly spaced frames** (section 6): keyframe timing follows visual change, not meaning, and the budget
(`min_gap_s=3`) is too small for MEVA's many half-second actions.

### Check yourself (stage 2)
1. Why does a 4×4 tile hash notice a small moving person when a single 64-bit hash doesn't?
2. What would `min_gap_s=1` do to the keyframe count and to the share of labels hit?
3. Why can evenly spaced frames beat gated keyframes on UCA, by construction?

---

## 3. Stage 3: events (`gpu_stages.py`, GPU)

Runs only inside segments (plus one idle frame per 5 min), on frames decoded at 4 per second and resized to
640 px. These frames are also stage 4's candidates. Three independent methods; a model that can't load is skipped
with a warning (`Models.failed`).

### 3.1 Detector + tracker → track events (`track_events`) — ran

- **YOLO-World** [9] (`yolov8s-worldv2.pt`) is an *open-vocabulary* detector: class names are given as text, so we
  can ask for "box" or "shopping cart" without retraining. 13 classes, confidence ≥ 0.15.
- **ByteTrack** [10] links detections across frames into tracks; it also uses low-confidence boxes in a second
  matching round, which keeps partly hidden people tracked.
- **Rules** on the tracks (v2's heuristics; distances in box heights): `<class>_appears` (track ≥ 1 s starts),
  `person_loitering` (≥ 20 s within 1.5 box heights), `running` (> 1.2 box heights/s for ≥ 0.75 s),
  `object_left` (bag/box ≥ 10 s, < 0.3 box heights of movement), `crowd` (≥ 5 people for ≥ 2 s).

**Run:** G421 15:55 produced **391 tracks** in 5 min with ~25 people visible; 328 of its 352 events are
`person_appears`, mostly the same people getting new IDs after occlusions. Track events are a weak signal in
crowds until tracking is tuned (supervisor's plan: January).

### 3.2 SigLIP frame tags and embeddings (`siglip_events`) — **failed to load**

**SigLIP** [11] (`google/siglip-base-patch16-224`) maps images and text into one vector space, like CLIP [12], but
is trained with a per-pair sigmoid loss, so each image-text score stands on its own. v2 compares every frame with a
12-event prompt bank and "nothing happening" prompts (event if the best event prompt beats the best background
prompt by > 1.0 logit). More importantly, **its 768-number image embeddings are the search index**.

**Run:** `SiglipTokenizer requires the SentencePiece library`. No tags, no embeddings, the same failure as in the
earlier café run. Fix: install `sentencepiece` and `protobuf` into the environment the kernel uses.

### 3.3 X-CLIP clip tags (`xclip_events`) — ran, nothing visible

**X-CLIP** [13] (`microsoft/xclip-base-patch32`) extends CLIP to 8-frame clips, so it can see motion over time.
Clips every second, scored against the prompt bank + "nothing is happening" with a softmax over 13 prompts; an
event needs probability > 0.35 and above background. This checkpoint was trained supervised on Kinetics-400 [14],
so its "zero-shot" scores on our prompts are less reliable than the name suggests, and with 13 prompts sharing
one softmax, 0.35 is rarely reached.

**Run:** loaded and ran; in the video inspected, all events are tracker events (café run: 0 X-CLIP events too).

### 3.4 Merging and importance (`merge_events`, `importance_curve`, `event_anchors`)

- Events with the same method, label and segment < 2 s apart are merged.
- **Support** = how many methods saw the same label overlapping in time. Agreement between independent methods is
  the main evidence an event is real. **Run:** with SigLIP down and X-CLIP silent, every event had support 1.
- **Importance** (0-1 per candidate frame): half from motion relative to the threshold, raised by events, 1.0
  around every track start, 0 for idle frames. With hundreds of track starts in a crowded clip, it is near 1
  almost everywhere there.
- **Anchors** (must be kept): peaks of events with support ≥ 2, any track event or score ≥ 0.7, and track starts.

### Check yourself (stage 3)
1. What makes YOLO-World "open-vocabulary" [9]?
2. What does ByteTrack's second matching round solve [10]? Why do IDs still switch in a crowd?
3. Why does SigLIP's sigmoid loss make a single score more meaningful than CLIP's softmax [11]?
4. Why is the X-CLIP checkpoint's zero-shot behaviour limited [13, 14]? What would a fairer test be?
5. What happens to importance and anchors when the tracker gives 391 IDs for 25 people?

---

## 4. Stage 4: which frames go to the captioner and the index (`gpu_stages.py`)

All samplers choose from the same candidates (stage 3's 4 frames/s inside segments + idle frames). Budget for the
embedding-based samplers: about 8 frames per active minute, scaled by mean importance, 2-40 per segment.

| Sampler | How it chooses | Origin | Run |
|---|---|---|---|
| `phash` | one global 64-bit hash, ≥ 10 differing bits and ≥ 1 s apart | v1 idea, v2 code | ✅ ran, **became the primary** |
| `novelty` | keep a frame if its SigLIP embedding is far from **every** frame kept so far in the segment | v2 heuristic | ⏭️ skipped (no embeddings) |
| `kcenter` (intended primary) | greedy **k-center** from the anchors, distance = 85 % SigLIP + 15 % time, weighted by importance | k-center greedy [15], used for core-sets [16]; weighting v2's | ⏭️ skipped |
| `importance` | budget spread along the importance curve (inverse-CDF), plus anchors, near-duplicates removed | v2 heuristic | ⏭️ skipped |
| `v1_keyframes` | stage 2's keyframes moved to the nearest candidate | merge | ✅ |

Greedy k-center picks points so that the largest distance from any point to its nearest chosen point is small;
Gonzalez [15] showed it is within a factor of 2 of optimal. Sener & Savarese [16] used it to pick a representative
subset of a dataset. Here: choose frames so no moment of a segment looks very different from every chosen frame.

**Metrics** (`evaluate_sampling`): coverage and redundancy in SigLIP space (NaN in this run), event and track
recall against the pipeline's **own** events (partly circular for samplers seeded with anchors), and, the fair
one, labels hit (`gt_eval_sampling.csv`). **Run, G421 15:55:** `phash` 72 frames, 60 % of labels; `v1_keyframes`
57 frames, 57 %.

**Hand-off** (`write_index_records`): `events/index_records.jsonl`, one record per frame of the primary sampler
(camera, video, frame, time, **wall-clock time**, segment, importance, events, objects, SigLIP tag,
`embedding_row`), and `events/index_embeddings.npy` (SigLIP vectors). **Run:** 1,075 records written from `phash`;
the embeddings file is empty and no stage-4 JPGs were saved (only `kcenter` exports them). The database can store
these records, but they can't be searched until SigLIP works.

### Check yourself (stage 4)
1. Explain k-center greedy in two sentences. What does the factor-2 guarantee in [15] mean?
2. Why does "far from every kept frame" avoid repeats that "far from the last kept frame" doesn't?
3. Why is `kcenter`'s event recall partly circular, and what should we report instead?

---

## 5. Evaluation and data

### 5.1 What the code measures (`motion_gating.gate_table`, `keyframe_hits`; `batch.run_cpu`)

For every video with labels, for every gate and keyframe set:
- **frame precision / recall / F1** (share of sampled moments correctly marked active), **event recall** (share of
  labelled items a segment covers by more than half, or with temporal IoU ≥ 0.3), % of video active;
- **labels hit**: share of labelled items with at least one keyframe inside. The number that matters downstream:
  an item without a keyframe is never captioned or indexed;
- two **baselines**: the same number of frames spread **evenly** over the video, and 1 frame/s. Gating only earns
  its place if it beats even spacing at the same budget.

### 5.2 Data

| Dataset | What it gives us | On disk (VM) | Ref |
|---|---|---|---|
| **UCA** (UCF-Crime Annotation) | the supervisor's main benchmark: 23,542 sentences with start/end times over 110.7 h of UCF-Crime video; text queries with timestamps, exactly what Recall@K needs | 166 dev videos, 9.0 h, all from the **val** split (test kept for final numbers) | [17] |
| UCF-Crime | the underlying videos, 13 anomaly types | (via UCA) | [18] |
| MEVA | multi-camera, untrimmed footage of an access-controlled site, scripted and spontaneous activity; activity labels, no text queries | 82 clips, 5.7 h, 4 cameras at one school site on 2018-03-13; 61 have activity labels | [19] |

**How labels are used (`datasets.py`).**
- **MEVA** `*.activities.yml` (KPF): one activity per line with a frame range; seconds = frame / fps. Only ~37
  activity types are labelled, so unlabelled motion is not an error: the labels say "these moments must be
  caught". Files are matched to videos by name without the `.r13` suffix.
- **UCA** sentences cover each video back to back, quiet parts included, so frame precision and recall against
  them mean little; "labels hit" is the useful number. Later the same sentences become the Recall@K queries.
- **UCF-Crime is partly not CCTV** (news and YouTube compilations with cuts, replays, several angles). A cut looks
  like whole-frame motion. **MEVA is the honest test for gating; UCA is the benchmark for retrieval.**

### 5.3 What the supervisor asked for (not built yet)

Retrieval Recall@1/5/10 at temporal IoU 0.3 and 0.5 on UCA test queries, against CLIP-only and SigLIP-only
baselines; 50-100 of our own queries on MEVA and our footage; query latency and hours indexed per GPU-hour; how
often the captioner mentions things that aren't there. Needs embeddings (SigLIP), then a text encoder and the
search; the database and API come after.

---

## 6. Results (19 Nov 2026 batch run, default settings)

38 videos, 2.00 h: MEVA 12 clips (3 time slots × 4 cameras, all labelled, 680 activities), UCA 26 videos
(2 per category, all labelled, 194 sentences).

### 6.1 Overview

| | MEVA | UCA |
|---|---|---|
| Share of video active (MOG2) | 42.7 % (per clip 12-68 %) | 65.9 % (per video 19-100 %) |
| Segments | 58 | 97 |
| Keyframes (per minute) | 302 (5.0) | 735 (12.2) |
| Fewer frames than 1 frame/s | 11.9× | 4.9× |
| **Labels with a keyframe** | **48.8 %** | **80.9 %** |
| Same number of frames, evenly spaced | 49.6 % | 93.8 % |
| Labelled events a segment covers | 63.8 % | 75.8 % |
| CPU stages 1-2 | 4.9 min per video-hour (12× real time) | 2.9 min (21×) |
| GPU stages 3-4 (without SigLIP, without captioning) | 4.0 min per video-hour (15×) | 3.2 min (19×) |
| Hand-off records | 290 | 785 |

### 6.2 Gates compared

| | MEVA labels hit / events covered | UCA labels hit / events covered |
|---|---|---|
| diff | **51.0 %** / **69.3 %** | **88.1 %** / 83.5 % |
| fused | 51.2 % / 67.6 % | 85.6 % / **86.1 %** |
| mog2 (in use) | 48.8 % / 63.8 % | 80.9 % / 75.8 % |

### 6.3 MEVA by activity type (share with a keyframe)

Long activities are caught, short ones are not: `person_talks_to_person` 86 % (149 labels),
`person_texts_on_phone` 71 %, `person_exits_scene_through_structure` 58 %, `person_opens_facility_door` 43 %,
`person_puts_down_object` 20 %, `person_closes_vehicle_door` 11 %, `person_opens_vehicle_door` 12 %,
`person_picks_up_object` **10 %** (69 labels).

### 6.4 What the results say

1. **The pipeline runs end to end at useful speed:** 2 h of video in 15 min on one L4, all four cameras on one
   wall clock, labels from both datasets, no failures.
2. **Gated keyframes are no better than evenly spaced frames yet.** MEVA: on the diagonal. UCA: below it (by
   design, since UCA describes quiet parts too). One frame per second reaches 93 % on the busiest MEVA clip. The
   budget, not the gate, is the main limit: gating should decide *where* to spend frames, not cut them to ~5 per
   minute.
3. **MEVA is busy, not empty.** This school site is active 43 % of the time on average. In the crowded indoor
   camera (G421) the adaptive threshold rises above real activity (section 1.6).
4. **Frame differencing beat MOG2** on both datasets.
5. **The GPU stages are not yet useful:** SigLIP didn't load (no embeddings), X-CLIP shows nothing, tracking
   gives ~390 IDs for ~25 people.

### 6.5 Earlier single-video results (before the merge; indicative only)

- **Café video (5 min, busy lobby):** v1 (greyscale MOG2, threshold from all samples) marked 27 % active and
  missed walking people; v2 (colour, quieter-half threshold, fused) marked 67 %. Against `gt_motion.csv`
  (**origin unknown**), only 41 % of events got a pHash keyframe. SigLIP also failed to load there; X-CLIP 0 events.
- **MEVA 30-min video (mostly idle outdoor):** 109 keyframes instead of 1,801 (16.5× fewer), v1 settings.

---

## 7. Weak spots and open questions

**To fix or check now**
1. Install `sentencepiece` + `protobuf` in the kernel's environment (Python 3.13, `awsenv`) and re-run the GPU
   phase: embeddings, `kcenter`, SigLIP tags.
2. G421 15:55: the motion score is exactly 0 for the last ~50 s while labelled activity continues. Frozen frames
   or a decoding problem? Open the video and check.

**To decide with experiments** (settings and expected effects: `docs/explanation.md`, "Suggested experiments")
3. Gate: switch to `diff` or `fused`?
4. Keyframe budget: `min_gap_s=1`, or "1 frame/s inside segments"? Must beat even spacing.
5. Busy scenes: `q_noise=0.2`, a lower `k`, or a per-camera fixed threshold?
6. Lighting filter: raise `mog2_illum_ncc` or turn it off on indoor cameras?
7. X-CLIP: margin over the background prompt instead of the 0.35 softmax threshold, or drop it?
8. Tracking: tune ByteTrack for crowds, or rely on SigLIP/VLM checks instead of track rules?
9. Process MEVA per camera as one continuous stream instead of separate 5-min files (MOG2 restarts)?

**For the report**
10. Where did the café `gt_motion.csv` come from? Until known, its numbers stay out of the report.
11. The deep gate stays off unless a test shows it is worth a full-video GPU pass.

---

## Section owners

| Section | Checked by | Date | Notes |
|---|---|---|---|
| 1. Motion gating (1.1-1.7, 1.9) | | | |
| 1.8 Deep gate | | | |
| 2. Keyframes | | | |
| 3. Events | | | |
| 4. Frame selection + hand-off | | | |
| 5-6. Evaluation, data and results | | | |

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
