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

**Status (26 Nov 2026, Practicum pilot).** Full run of every stage on the team VM (one L4), from a clean start:
38 videos, 2.00 h (1 h MEVA, 1 h UCA), no video failed. CPU phase 7.1 min, GPU phase (stages 3-4, with SigLIP)
8.8 min, clip captions 27.2 min: **43 min in total**, about 3× faster than real time. 167 clip captions, 2,673
frames with SigLIP embeddings; frame captions were stopped by hand after 220 frames. The search was then built and
scored against the CLIP and SigLIP baselines on 218 queries. Results: section 6; the notebooks with all their
outputs: `runs/2026-11-26/`.

The first run (19 Nov) covered stages 1-4 only. Its gating and keyframe numbers are identical to the 26 Nov run
(same settings), but SigLIP failed to load there (missing `sentencepiece`), which is fixed now.

---

## 0. The big picture

A fixed camera records long stretches where little happens. Captioning or indexing every frame is slow and
expensive, and most frames repeat the same scene. The pipeline narrows the video down in four stages, each more
expensive than the last but running on less of the video, and then describes what is left in words (stage 5):

```
video (25-30 frames/s)
  │
  ├─ Stage 1  motion gating        5 samples/s, cheap (CPU)        → segments: time ranges with activity
  ├─ Stage 2  keyframes            inside segments, cheap (CPU)    → a few representative frames per segment
  ├─ GT check                      CPU                             → did segments / keyframes catch the labels?
  ├─ Stage 3  events               inside segments, 4 frames/s, GPU→ who/what is there, what happens
  ├─ Stage 4  frame selection      same candidates, GPU            → frames for the captioner + search index
  │                                                                  (index_records.jsonl + embeddings)
  └─ Stage 5  captions             stage 4's frames + clips, GPU   → structured text per frame and per clip
                                                                     (captions/captions.csv)
then
  Search                           in memory, per query            → ranked clips with start/end and a 10-s window
```

| Part | Code | Origin |
|---|---|---|
| Dataset discovery, selection, labels | `rewind/data.py` | added for the batch run |
| Batch loop (phases cpu → gpu → vlm_clips → vlm_frames, pre-flight check, fresh start, resume, per-video errors) | `batch.py`, `1_pipeline.ipynb` | added for the batch run |
| Stage 1 | `motion_gating.py` (+ `gpu_stages.scan_deep` for the deep gate) | v1 (Farid), with v2's threshold, colour MOG2 and fused vote |
| Stage 2 | `keyframes.py` | v1 |
| Stages 3, 4 | `gpu_stages.py` | v2 (Muzayyan, written with an LLM) |
| Stage 5 (captions) | `captions.py`, phases `vlm_clips` / `vlm_frames` | added 26 Nov 2026 (written with an LLM) |
| Caption quality checks | `caption_eval.py` | added 26 Nov 2026 (written with an LLM) |
| Search and its evaluation | `search.py`, `2_search.ipynb` | added 26 Nov 2026 (written with an LLM) |
| Results views | `3_explore.ipynb` | added for the batch run |

The batch runs stages 1-2 (and the ground-truth check) for **every video first**, then stages 3-4 for every
video, then clip captions, then frame captions, so a GPU problem never costs the earlier results.

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

**Run (26 Nov):** 4.1 min of decoding per hour of MEVA (≈15× real time), 2.5 min per hour of UCA; with
keyframes and the ground-truth check, stages 1-2 take 4.4 and 2.7 min per video-hour.

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
(shown yellow in `3_explore.ipynb`). Flat, low-texture regions correlate easily with the background, so the
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
hash per tile of a 4×4 grid (`tile_phash`), distance = **mean** over the 16 tiles (`change_threshold=6` bits).
The mean (not the max) is on purpose: flat tiles (sky, grass) flip a few bits from compression noise alone, and
the max would pick that up. The price: a change in a single tile is divided by 16, so one small person crossing a
wide scene often stays below the threshold; the tiles help mostly when several parts of the frame change.

**Run:** 302 keyframes for 1 h of MEVA (5 per minute, 12× fewer than 1 frame/s), 735 for 1 h of UCA (12 per minute,
5× fewer). In busy videos almost all are `change` keyframes. **They caught no more labelled moments than the same
number of evenly spaced frames** (section 6): keyframe timing follows visual change, not meaning, and the budget
(`min_gap_s=3`) is too small for MEVA's many half-second actions.

### Check yourself (stage 2)
1. With the distance averaged over 16 tiles, how many bits must one tile change to reach `change_threshold=6` on
   its own? What does that say about small moving people?
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

**Run:** G421 15:55 produced **391 tracks** in 5 min with ~25 people visible; 328 of its 380 events are
`person_appears`, mostly the same people getting new IDs after occlusions. Track events are a weak signal in
crowds until tracking is tuned (supervisor's plan: January).

### 3.2 SigLIP frame tags and embeddings (`siglip_events`) — ran

**SigLIP** [11] (`google/siglip-base-patch16-224`) maps images and text into one vector space, like CLIP [12], but
is trained with a per-pair sigmoid loss, so each image-text score stands on its own. v2 compares every frame with a
12-event prompt bank and "nothing happening" prompts (event if the best event prompt beats the best background
prompt by > 1.0 logit). More importantly, **its 768-number image embeddings are the search index**.

**Run (26 Nov):** loaded and ran on all 38 videos after `sentencepiece` was installed in the kernel's environment
(on 19 Nov it failed with `SiglipTokenizer requires the SentencePiece library`). Every video now has embeddings.
The zero-shot tags are unreliable in crowds: in the calm school foyer of G421 15:55 the prompt-bank tags reported
`fighting` 14 times.

### 3.3 X-CLIP clip tags (`xclip_events`) — ran, rarely fires

**X-CLIP** [13] (`microsoft/xclip-base-patch32`) extends CLIP to 8-frame clips, so it can see motion over time.
Clips every second, scored against the prompt bank + "nothing is happening" with a softmax over 13 prompts; an
event needs probability > 0.35 and above background. This checkpoint was trained supervised on Kinetics-400 [14],
so its "zero-shot" scores on our prompts are less reliable than the name suggests, and with 13 prompts sharing
one softmax, 0.35 is rarely reached.

**Run:** loaded and ran. On 19 Nov (without SigLIP) all events in the video inspected were tracker events (café
run: 0 X-CLIP events too). On 26 Nov the prompt-bank keys (`fighting`, `person_walking`, `door_interaction`) appear
in G421 15:55, but SigLIP and X-CLIP share the same keys; check the `source` column of `events.csv` before crediting
X-CLIP.

### 3.4 Merging and importance (`merge_events`, `importance_curve`, `event_anchors`)

- Events with the same method, label and segment < 2 s apart are merged.
- **Support** = how many methods saw the same label overlapping in time. Agreement between independent methods is
  the main evidence an event is real. (19 Nov, with SigLIP down and X-CLIP silent, every event had support 1;
  not yet re-counted for the 26 Nov run.)
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

| Sampler | How it chooses | Origin | Run, G421 15:55 (26 Nov): frames, labels hit |
|---|---|---|---|
| `phash` | one global 64-bit hash, ≥ 10 differing bits and ≥ 1 s apart | v1 idea, v2 code | 72, 60 % (fallback primary) |
| `novelty` | keep a frame if its SigLIP embedding is far from **every** frame kept so far in the segment | v2 heuristic | 10, 29 % |
| `kcenter` (**primary**) | greedy **k-center** from the anchors, distance = 85 % SigLIP + 15 % time, weighted by importance | k-center greedy [15], used for core-sets [16]; weighting v2's | **315, 74 %** |
| `importance` | budget spread along the importance curve (inverse-CDF), plus anchors, near-duplicates removed | v2 heuristic | 53, 53 % |
| `v1_keyframes` | stage 2's keyframes moved to the nearest candidate | merge | 57, 57 % |

Greedy k-center picks points so that the largest distance from any point to its nearest chosen point is small;
Gonzalez [15] showed it is within a factor of 2 of optimal. Sener & Savarese [16] used it to pick a representative
subset of a dataset. Here: choose frames so no moment of a segment looks very different from every chosen frame.

**Metrics** (`evaluate_sampling`): coverage and redundancy in SigLIP space, event and track recall against the
pipeline's **own** events (partly circular for samplers seeded with anchors), and, the fair one, labels hit
(`gt_eval_sampling.csv`).

**`kcenter` floods crowded scenes.** It starts from **every** event anchor, and the anchors are not capped by the
budget. In G421 15:55 the tracker's ~390 track starts became anchors, so `kcenter` kept 315 frames in 5 min (63 per
minute, more than 1 frame/s: no saving). Its 74 % labels hit is bought with that budget. Across the pilot set the
hand-off holds 2,673 frames against 1,037 stage-2 keyframes. Fix (January): cap the anchors at the budget.

**Hand-off** (`write_index_records`): `events/index_records.jsonl`, one record per frame of the primary sampler
(camera, video, frame, time, **wall-clock time**, segment, importance, events, objects, SigLIP tag,
`embedding_row`), and `events/index_embeddings.npy` (SigLIP vectors). **Run (26 Nov):** 2,673 records from
`kcenter` (1,631 MEVA, 1,042 UCA), all with embeddings; `kcenter`'s JPGs in `events/keyframes/kcenter/`. The
search index (section 4c) reads these files directly; the Capstone database will ingest the same records.

### Check yourself (stage 4)
1. Explain k-center greedy in two sentences. What does the factor-2 guarantee in [15] mean?
2. Why does "far from every kept frame" avoid repeats that "far from the last kept frame" doesn't?
3. Why is `kcenter`'s event recall partly circular, and what should we report instead?

## 4b. Stage 5: captions (`captions.py`, GPU) — clip captions ran 26 Nov, frame captions partly

**Goal:** text a search can match ("person in a red jacket carrying a box near the door"), tied to an exact
camera, time and clip. A vision-language model (VLM) looks at the frames stage 4 chose and writes it.

**Model.** Qwen2.5-VL-7B-Instruct [20] by default (the supervisor's "small Qwen-VL"), run through Hugging Face
`transformers`, bf16, greedy decoding (the same frame always gets the same caption). It follows the LLaVA
template from the reading list (vision encoder → projector → LLM) and adds native dynamic resolution: an image
becomes a variable number of 28×28-pixel tokens instead of being squashed to a fixed size, which matters for
small, far-away people. Qwen3-VL checkpoints load through the same code (`model` setting).

**Two levels.**

| Level | Input | Output | Why |
|---|---|---|---|
| frame | one selected frame (stage 4's hand-off, or stage 2's keyframes if stage 4 hasn't run), up to ~1 MP | JSON: caption, scene, lighting, people count, people (appearance / action / where, at most 6), vehicles (type / colour / action, at most 6), objects, notable, uncertain | fine-grained "who is there and what do they look like"; shares `embedding_row` with the SigLIP vector of the same frame, so text and image search can be combined per frame |
| clip | a motion segment cut into pieces of ≤ 60 s; up to 6 frames of it in time order, each labelled "+12.3 s", ~0.35 MP each | JSON: summary, actions, largest number of people in one frame, people (short appearance descriptions), vehicles, objects, notable, uncertain (no list limits yet) | actions over time (enters, picks up, leaves) that one frame can't show; a clip has a start and end, so it is what a search returns and the player jumps to |

Clip frames are the stage-2/4 frames inside the clip, thinned evenly to 6; if fewer than 3 were selected, evenly
spaced frames from the video fill it up. The 60-s cap matters for UCA, where one segment can cover the whole
video, and keeps clips close to the length of the UCA sentences we will score with temporal IoU.

**Not making things up.** Results are evidence, so the system prompt says: describe only what is clearly
visible, never guess identities or intentions, put doubts in `uncertain`. The detector's events and objects are
stored next to each caption but **not** shown to the VLM, so the two stay independent and can check each other
later (the Capstone's verification step). How often the captions still mention things that aren't there is
measured in three ways (`caption_eval.py`, `3_explore.ipynb` section 6): by hand on 30 random clips split over 5
people (hallucination and miss rate); automatically on every clip against the detector's tracks (mentioned but
never tracked / tracked but not mentioned, people count); and against the ground-truth labels (sentence-embedding
similarity to the UCA sentence of the same seconds, against other videos' captions as a baseline).

**Outputs** (`captions/`): `frame_captions.jsonl`, `clip_captions.jsonl` (full JSON + raw answer + timing),
`captions.csv` (both levels in one table: camera, start/end, wall clock, text, `search_text`, preview image),
`frames/*.jpg` (the frames at the size the model saw). Per video `summary_vlm.json`: counts, unparsed answers,
seconds per frame / clip, minutes per video-hour, peak VRAM.

**Run (26 Nov, one L4, Qwen2.5-VL-7B, bf16).**

| | MEVA | UCA |
|---|---|---|
| Clips captioned | 64 of 64 | 103 of 103 |
| Answers that were not valid JSON | 2 | 1 |
| Seconds per clip (mean) | 10.9 | 8.8 |
| Clip captions per video-hour | 12.0 min | 14.4 min |
| Peak GPU memory | 19.8 GB | 17.4 GB |
| Clips flagged as unusual (`notable`) | 0 | 4, all correct (2 fights, a dust cloud, an explosion flash) |
| Frame captions (stopped by hand) | 220 of 1,631; 3.8 s per frame at batch 8 | 0 of 1,042 |

- **Frames are too slow to caption all of them:** 2,673 frames × 3.8 s ≈ 2.8 GPU-hours for 2 h of video, slower
  than real time. Clips are the default and frames optional.
- **Two GPUs** (`VLM_DEVICE = ["cuda:0", "cuda:1"]`, now the default) are implemented but not yet timed; the
  26 Nov run used one.
- **Quality:** captions get the gist of crowded scenes (people entering and leaving through doors, sitting at
  tables), but one G421 clip lists "man in a black jacket; woman in a black jacket" 19 times (a greedy-decoding
  loop: needs a repetition penalty and list limits in the clip prompt), one says the scene "remains unchanged" while
  the room empties, and some details ("a purple jacket with a number 14") can't be confirmed at this resolution.
- **`notable` is precise but rarely used:** only 4 of the 24 crime and accident videos got a flag; no arson,
  robbery, shooting, shoplifting, stealing or vandalism clip did.

**Caption-quality checks (26 Nov, the two automatic ones):**

| | MEVA (64 clips) | UCA (103 clips) |
|---|---|---|
| Clips with a likely hallucination (caption names a category the detector never tracked) | 15.6 % | 26.2 % |
| Clips with a likely miss (tracked but not mentioned) | 43.8 % | 28.2 % |
| People count within ±25 % of the detector | 59.7 % | 85.3 % |
| Labels whose covering caption beats other videos' captions (sentence similarity) | 64.0 % of 680 | 83.1 % of 194 |
| Caption-only search on the labels: top-5 has a covering clip | 17.3 % | 35.5 % |

The detector check is a screen, not ground truth: all ten MEVA flags are bags, which the detector may simply have
missed, and it knows nothing about actions or colours. The hand check (30 clips, 6 per person,
`outputs/batch/clip_review/`) has not been filled in yet; it gives the real hallucination rate and says how far to
trust the screen.

### Check yourself (stage 5)
1. Why are the detector's events not put into the prompt, even though they would make captions "better"?
2. A clip caption says "a man picks up a bag". Which file tells you which frames it saw, and how do you check it?
3. Why does `frame_max_pixels` matter more for MEVA than for a close-up indoor camera?

---

## 4c. Search (`search.py`, `2_search.ipynb`) — ran 26 Nov

**Goal:** a typed sentence returns ranked moments (video, camera, start-end, best frame), and a fair score for it.
Everything is kept in memory (≈ 2 h of video = a few thousand vectors), so no vector database or server is needed
yet; the same vectors move to one later.

| Method | Query side | Index side | Returns |
|---|---|---|---|
| `captions` | `bge-small-en-v1.5` [21] with its retrieval prefix | the clip caption's summary, actions, people, vehicles, objects, notable (not `uncertain`) | clips |
| `siglip_selected` | SigLIP text tower | stage 4's SigLIP frame embeddings (768-d); one result per 10 s per video | frames → their clip |
| `hybrid` | both | reciprocal rank fusion [22], score = Σ 1/(60 + rank); a clip's SigLIP score = its best frame | clips |
| `siglip_1fps`, `clip_1fps` | SigLIP / CLIP ViT-B/32 text | 1 frame/s of every **whole** video (no gating, no captions) | frames |

All methods take the same filters: dataset, video and camera (`SearchIndex.search`). There is **no time-of-day
filter yet** (Capstone, with the database). The two baselines are what the supervisor asked for ("CLIP-only and
SigLIP-only"): they see every second of video, so they show what gating + captions add or lose.

**Spans.** A result has its clip (≤ 60 s) and a 10-s window (`moment_s`) around the best SigLIP frame inside the
clip, kept inside the clip; a clip of 10 s or less is its own window. The window is placed this way for **every**
clip method, including `captions`, so caption-only IoU numbers also depend on SigLIP. A 5-s event inside a 60-s
clip has IoU 0.08, so IoU scores use the window; the lenient score uses the clip.

**Queries.** UCA: every sentence of the selected videos, answer = its own span (a few hundred queries). MEVA: one
operator-style sentence per activity type (`search.MEVA_QUERIES`, e.g. *"a person gets out of a car"*), answers =
every labelled instance on any camera, so a hit on any one counts. Plus 20 free-text operator queries without
ground truth for the demo (objective 5).

**Metrics.** Recall@1/5/10 at temporal IoU 0.3 / 0.5, lenient Recall@K (any overlap), MRR at IoU 0.3, per dataset ×
method × setting (`corpus`: all videos of the dataset; `video`: only the query's video, UCA's grounding setting).
**reachable %** = queries whose answer overlaps some clip: the ceiling for the gated methods (the gating ceiling,
§6: MEVA 64 %, UCA 76 % of labelled items covered by a segment), and lenient R@5 over reachable queries only separates
gating misses from search misses. Latency = encode + rank one query, median and p95. Not implemented yet: bootstrap
confidence intervals (planned for the final numbers) and a score threshold that lets the search answer "nothing
found". Results of the 26 Nov run: section 6.5.

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

**How labels are used (`rewind/data.py`).**
- **MEVA** `*.activities.yml` (KPF): one activity per line with a frame range; seconds = frame / fps. Only ~37
  activity types are labelled, so unlabelled motion is not an error: the labels say "these moments must be
  caught". Files are matched to videos by name without the `.r13` suffix.
- **UCA** sentences cover each video back to back, quiet parts included, so frame precision and recall against
  them mean little; "labels hit" is the useful number. Later the same sentences become the Recall@K queries.
- **UCF-Crime is partly not CCTV** (news and YouTube compilations with cuts, replays, several angles). A cut looks
  like whole-frame motion. **MEVA is the honest test for gating; UCA is the benchmark for retrieval.**

### 5.3 What the supervisor asked for

Retrieval Recall@1/5/10 at temporal IoU 0.3 and 0.5 on UCA queries, against CLIP-only and SigLIP-only baselines
(section 4c, `2_search.ipynb`; on UCA **val** videos, test kept for final numbers); our own queries on MEVA (one per
activity type + 20 free-text); query latency (4c) and hours indexed per GPU-hour (`3_explore.ipynb` section 7); how
often the captioner mentions things that aren't there (section 4b, hand check). Still missing: the database, the
API, the connection to the dashboard, time-of-day filters, and an LLM layer that rewrites queries (dataset, video and
camera filters exist in `SearchIndex.search`).

---

## 6. Results (26 Nov 2026 pilot run, default settings)

38 videos, 2.00 h: MEVA 12 clips (3 time slots × 4 cameras, all labelled, 680 activities), UCA 26 videos
(2 from each of 12 crime categories + 2 normal videos; Arrest has only one val video and was left out; all
labelled, 194 sentences). Stages 1-2 give the same numbers as the 19 Nov run (same settings); times are from the
26 Nov run, which started without any cache.

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
| CPU stages 1-2 | 4.4 min per video-hour (14× real time) | 2.7 min (22×) |
| GPU stages 3-4 (with SigLIP) | 4.9 min per video-hour (12×) | 3.6 min (17×) |
| Hand-off frames (`kcenter`) | 1,631 | 1,042 |
| Clip captions | 12.0 min per video-hour (5×) | 14.4 min (4×) |
| **All stages 1-5 with clip captions** | **21.3 min per video-hour (2.8×)** | **20.7 min (2.9×)** |

### 6.2 Gates compared

| | MEVA labels hit / events covered | UCA labels hit / events covered |
|---|---|---|
| diff (313 MEVA / 864 UCA keyframes) | **51.0 %** / **69.3 %** | **88.1 %** / 83.5 % |
| fused (296 / 818) | 51.2 % / 67.6 % | 85.6 % / **86.1 %** |
| mog2, in use (302 / 735) | 48.8 % / 63.8 % | 80.9 % / 75.8 % |

`diff` and `fused` also keep more keyframes on UCA (18 % and 11 % more), so part of their gain there is budget.

### 6.3 MEVA by activity type (share with a keyframe)

Long activities are caught, short ones are not: `person_talks_to_person` 86 % (149 labels),
`person_texts_on_phone` 71 %, `person_exits_scene_through_structure` 58 %, `person_opens_facility_door` 43 %,
`person_puts_down_object` 20 %, `person_closes_vehicle_door` 11 %, `person_opens_vehicle_door` 12 %,
`person_picks_up_object` **10 %** (69 labels).

### 6.4 What the indexing results say

1. **The pipeline runs end to end:** 2 h of video through all five stages in 43 min on one L4, all four MEVA
   cameras on one wall clock, labels from both datasets, no failures. At about 3× real time it is **short of the 4×
   target**; captioning is 56 % (MEVA) to 70 % (UCA) of the time, so the untimed two-GPU captioning matters most.
2. **Gated keyframes are no better than evenly spaced frames yet.** MEVA: on the diagonal. UCA: below it (by
   design, since UCA describes quiet parts too). One frame per second reaches 93 % on the busiest MEVA clip. The
   budget, not the gate, is the main limit: gating should decide *where* to spend frames, not cut them to ~5 per
   minute.
3. **MEVA is busy, not empty.** This school site is active 43 % of the time on average. In the crowded indoor
   camera (G421) the adaptive threshold rises above real activity (section 1.6).
4. **Frame differencing beat MOG2** on both datasets, partly with a larger budget (6.2).
5. **The GPU stages work but are noisy in crowds:** tracking gives ~390 IDs for ~25 people, the zero-shot tags
   report "fighting" in a calm foyer, and `kcenter` keeps more than 1 frame/s there (section 4).
6. **Clip captions are usable but imperfect** (section 4b): 98 % valid JSON, correct gist, some loops and
   unverifiable details; the hand check is still to do.

### 6.5 Search results (218 queries)

% of queries with a correct result; search over all videos of the dataset unless stated. IoU is scored on the
10-s window around each clip's best SigLIP frame (also for `captions`).

| Method | UCA R@1 IoU 0.3 | UCA R@5 IoU 0.3 | UCA R@5 IoU 0.5 | UCA R@5 lenient | UCA own video: R@5 IoU 0.3 | MEVA R@5 lenient |
|---|---|---|---|---|---|---|
| captions | 6.7 | 14.4 | 7.7 | 32.5 | 35.1 | 62.5 |
| siglip_selected | 6.2 | 17.0 | 11.3 | 33.0 | 40.7 | 50.0 |
| **hybrid** | **7.7** | **19.6** | **12.4** | **42.3** | 35.1 | **79.2** |
| siglip_1fps (baseline) | 5.7 | 19.1 | 13.4 | 27.3 | 44.8 | 33.3 |
| clip_1fps (baseline) | 5.2 | 18.0 | 10.8 | 22.2 | 47.9 | 29.2 |
| reachable (answer overlaps a clip) | 88.7 | 88.7 | 88.7 | 88.7 | 88.7 | 95.8 |

1. **Hybrid finds the right clip far more often than the baselines** (UCA lenient R@5 42.3 % vs 27.3 % and
   22.2 %; MEVA 79.2 % vs 33.3 % and 29.2 %). Each half alone reaches about 33 % on UCA; fusing adds about 10
   points, so captions and SigLIP make different mistakes. Not yet tested with confidence intervals.
2. **It is no better at the exact moment.** At IoU 0.3 hybrid (19.6 %) is level with the baselines (19.1 %,
   18.0 %), and within one known video the baselines win (44.8 %, 47.9 %): they have a frame every second, we have
   a few per clip. Better span boundaries are the main work left (Capstone, after TFVTG).
3. **Most misses are search misses:** of all UCA queries, 11 % have no indexed clip at all (gating), 46 % have one
   that is ranked too low. UCA sentences are long and specific; our captions are more general.
4. **Two scores flatter the results.** MEVA labels are dense (149 instances of talking, 69 of picking up in 1 h),
   so MEVA lenient is an upper bound; a UCA video has only ~4 clips, so the single-video lenient score is close to
   the reachable share for every method.
5. **Search is fast:** median 36 ms per query (p95 37-63 ms across methods), index built in 6 s from the saved
   outputs, baseline embeddings (7,239 frames) in 3.2 min.
6. **The 20 operator queries look plausible, with one gap.** "A fight between two people", "people walking into a
   building", "a dog" and "a person breaking into a car" return fitting clips; "smoke or fire" returns the explosion
   clip the VLM flagged. But the search always returns something: "a person on a bicycle" and "two people shaking
   hands" both return a crowd around a person on the ground. A score threshold for "nothing found" is needed.

### 6.6 Earlier single-video results (before the merge; indicative only)

- **Café video (5 min, busy lobby):** v1 (greyscale MOG2, threshold from all samples) marked 27 % active and
  missed walking people; v2 (colour, quieter-half threshold, fused) marked 67 %. Against `gt_motion.csv`
  (**origin unknown**), only 41 % of events got a pHash keyframe. SigLIP also failed to load there; X-CLIP 0 events.
- **MEVA 30-min video (mostly idle outdoor):** 109 keyframes instead of 1,801 (16.5× fewer), v1 settings.

---

## 7. Weak spots and open questions

**Done since 19 Nov:** SigLIP loads (`sentencepiece` in the kernel's environment), `kcenter` is the primary
sampler again, clip captions ran on every video, the search ran against both baselines.

**To fix or check now**
1. Fill in the caption hand check (30 clips, 6 per person, `outputs/batch/clip_review/`).
2. Cap `kcenter`'s event anchors at the frame budget (section 4).
3. Clip captions: repetition penalty and list limits in `CLIP_PROMPT` (the 19-times loop, section 4b).
4. Time captioning on both GPUs (`VLM_DEVICE = ["cuda:0", "cuda:1"]`): does it reach the 4× real-time target?
5. Search: a score threshold for "nothing found"; span edges from frame-level similarity (TFVTG-style) for IoU.
6. G421 15:55: the motion score is exactly 0 for the last ~50 s while labelled activity continues. Frozen frames
   or a decoding problem? Open the video and check.

**To decide with experiments** (settings and expected effects: `docs/explanation.md`, "Suggested experiments")
7. Gate: switch to `diff` or `fused`? Compare at the same keyframe budget.
8. Keyframe budget: `min_gap_s=1`, or "1 frame/s inside segments"? Must beat even spacing.
9. Busy scenes: `q_noise=0.2`, a lower `k`, or a per-camera fixed threshold?
10. Lighting filter: raise `mog2_illum_ncc` or turn it off on indoor cameras?
11. X-CLIP: margin over the background prompt instead of the 0.35 softmax threshold, or drop it?
12. Tracking: tune ByteTrack for crowds, or rely on SigLIP/VLM checks instead of track rules?
13. Process MEVA per camera as one continuous stream instead of separate 5-min files (MOG2 restarts)?

**For the report**
14. Where did the café `gt_motion.csv` come from? Until known, its numbers stay out of the report.
15. The deep gate stays off unless a test shows it is worth a full-video GPU pass.

---

## Section owners

| Section | Checked by | Date | Notes |
|---|---|---|---|
| 1. Motion gating (1.1-1.7, 1.9) | | | |
| 1.8 Deep gate | | | |
| 2. Keyframes | | | |
| 3. Events | | | |
| 4. Frame selection + hand-off | | | |
| 4b. Captions and caption checks | | | |
| 4c. Search | | | |
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
20. S. Bai et al. (Qwen Team), "Qwen2.5-VL Technical Report," 2025. [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
21. S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, J.-Y. Nie, "C-Pack: Packed Resources for General Chinese Embeddings" (BGE), *SIGIR*, 2024. [doi:10.1145/3626772.3657878](https://doi.org/10.1145/3626772.3657878)
22. G. V. Cormack, C. L. A. Clarke, S. Buettcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," *SIGIR*, 2009. [doi:10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114)
