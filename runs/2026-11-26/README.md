# Run of 26 Nov 2026: first full run with captions

`pipeline_run.ipynb`, `search_run.ipynb` and `explore_run.ipynb` are the notebooks **with their outputs**, as run
on the VM (one L4, Qwen2.5-VL-7B-Instruct, `selection.csv` = 12 MEVA videos (1.0 h) + 26 UCA val videos (1.0 h)).
They were run before the repository was reorganised, so their import lines (`import batch`, `import search`, ...)
are the old ones, and the paths in their outputs show the VM folder under the repository's old name
(`/mnt/5T_storage/cctv-motion-gating`; the repository is now `REWIND-indexing-pipeline`). The clean, runnable
versions are `notebooks/1_pipeline.ipynb`, `2_search.ipynb` and `3_explore.ipynb`.

## What ran

| Phase | Time | Result |
|---|---|---|
| CPU (stages 1-2) | 7.1 min | 38/38 videos; MEVA 43% active, 302 keyframes (12x fewer than 1 fps); UCA 66% active, 735 keyframes (5x) |
| GPU (stages 3-4) | 8.8 min | 38/38 videos, **SigLIP embeddings for all 38** (sentencepiece fixed); 2,673 index records |
| Clip captions | 27.2 min | 167 clips (64 MEVA, 103 UCA), 3 not valid JSON; ~10 s per clip; peak VRAM 19.8 GB |
| Frame captions | stopped by hand | 220 of 2,673 frames (2 MEVA videos); 3.8 s per frame at batch 8 |
| Search (`search_run.ipynb`) | 3.3 min | index of 167 clips + 2,673 frames; baselines embedded 7,239 frames (1/s) in 3.2 min; 218 queries x 5 methods scored in 27 s |

Indexing cost at clip level on one L4: about 21 min of compute per hour of video (CPU 4.4 + GPU 4.9 + clips 12),
so roughly 3 hours of video per compute-hour.

## What looks good

- End to end on real data with no failed video.
- The VLM flagged 4 clips as unusual, all correct: two fights (Fighting045, Fighting049), an explosion's dust cloud
  (Explosion050) and an explosion flash (Explosion042).
- Clip captions get the gist of crowded scenes (people entering / leaving through doors, sitting at tables).

## Problems found (to fix or report)

1. **Keyframes are not better than uniform sampling at the same count.** Labelled moments with a keyframe:
   MEVA 48.8% vs 49.6% uniform; UCA 80.9% vs 93.8% uniform. And 36% of MEVA / 24% of UCA labelled events fall
   outside every motion segment, so no caption can find them. Short actions are the weak spot (MEVA
   `person_picks_up_object` 10%, `person_puts_down_object` 20%). The `diff` and `fused` gates cover more
   (MEVA 69% / 68%, UCA 84% / 86% of events) than `mog2` (64% / 76%) at a similar number of keyframes.
2. **Stage 4 hands off far too many frames in busy scenes:** G421 got 315 index records for 5 minutes (more than
   one per second, 0.95x "reduction"). Cause: k-center starts from every event anchor and the anchors aren't
   capped by the budget, and `person_appears` fires hundreds of times in crowds (tracker ID switches). This is why
   frame captions would take hours.
3. **Repetition in crowded clips:** one G421 clip lists "man in a black jacket; woman in a black jacket" 19 times,
   and repeats the jackets under `objects`. Typical greedy-decoding loop; needs a repetition penalty, a cap on list
   length and de-duplication.
4. **Details that may be invented:** e.g. "a purple jacket with a number 14", which can't be confirmed at this
   resolution. That's what the hand check is for.
5. **Hand check not done yet:** `explore_run.ipynb` made the clip sheets (`outputs/batch/clip_review/`, 30 clips
   over 5 people), but nobody has filled them in. The two automatic checks did run: likely hallucination in 15.6 %
   of MEVA and 26.2 % of UCA clip captions (detector cross-check; all MEVA flags are bags, so it is a screen).
6. **Recall of the "notable" field is low:** 4 of the 24 non-normal UCA videos were flagged (no arson, robbery,
   shooting, shoplifting, stealing or vandalism clip was).

## Search (218 queries: 194 UCA sentences, 24 MEVA activity types)

| Method | UCA R@5 IoU 0.3 | UCA R@5 lenient | UCA own video R@5 IoU 0.3 | MEVA R@5 lenient |
|---|---|---|---|---|
| captions | 14.4 | 32.5 | 35.1 | 62.5 |
| siglip_selected | 17.0 | 33.0 | 40.7 | 50.0 |
| **hybrid** | **19.6** | **42.3** | 35.1 | **79.2** |
| siglip_1fps (baseline) | 19.1 | 27.3 | 44.8 | 33.3 |
| clip_1fps (baseline) | 18.0 | 22.2 | 47.9 | 29.2 |

Reachable (an answer overlaps an indexed clip): UCA 88.7%, MEVA 95.8%. Hybrid finds the right clip far more often
than the baselines but is no better at the exact moment (and worse within one video, where the baselines have a
frame every second). UCA misses (hybrid): 46% search, 11% gating. Median 36 ms per query. MEVA lenient is inflated by
dense labels. No "nothing found": unrelated operator queries still return a clip.
