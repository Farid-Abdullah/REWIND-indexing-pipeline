# Run of 26 Nov 2026: first full run with captions

`pipeline_run.ipynb` and `explore_run.ipynb` are the two notebooks **with their outputs**, as run on the VM
(one L4, Qwen2.5-VL-7B-Instruct, `selection.csv` = 12 MEVA videos (1.0 h) + 26 UCA val videos (1.0 h)).
The clean, runnable versions are `pipeline.ipynb` / `explore.ipynb` in the repo root.

## What ran

| Phase | Time | Result |
|---|---|---|
| CPU (stages 1-2) | 7.1 min | 38/38 videos; MEVA 43% active, 302 keyframes (12x fewer than 1 fps); UCA 66% active, 735 keyframes (5x) |
| GPU (stages 3-4) | 8.8 min | 38/38 videos, **SigLIP embeddings for all 38** (sentencepiece fixed); 2,673 index records |
| Clip captions | 27.2 min | 167 clips (64 MEVA, 103 UCA), 3 not valid JSON; ~10 s per clip; peak VRAM 19.8 GB |
| Frame captions | stopped by hand | 220 of 2,673 frames (2 MEVA videos); 3.8 s per frame at batch 8 |

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
5. **No hand-check sheet yet:** it samples frame captions, and the frame phase was stopped.
6. **Recall of the "notable" field is low:** 4 of the 24 non-normal UCA videos were flagged (no arson, robbery,
   shooting, shoplifting, stealing or vandalism clip was).
