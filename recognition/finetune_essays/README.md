# Essay fine-tune (recognizer)

Light fine-tune of the recognizer on the hand-labelled essays (ideal per-line polygons +
transcriptions in PaddleOCR `labels.txt`). Separate from the main fine-tune, but reuses the
shared stack (model builder, processor, `TrOCRCollator`, `trainer_v2`, metrics, `JpegArtifacts`).

What's new here (the rest is reused, not re-implemented):

- **`geometry.py`** — rectify a 4-point line polygon to an upright crop (`warp_crop`), grow/shrink
  the box along its own axes (`expand_quad`).
- **`dataset.py`** — `EssayLineDataset`: reads `labels.txt`, warps each line **on the fly from the
  full page**, and in train **jitters the frame ±`frame_jitter`** (≈±2%: looser/tighter than the
  ideal box) so the recognizer tolerates real detector boxes. Split is **by page** (no leakage).
- **`augment.py`** — `EssayAugmenter`: shadows, mild colour shift, weak blur, quality drop
  (downscale), JPEG artifacts. Geometry stays in the dataset (it needs page pixels).

## Run

```bash
python finetune_essays/run.py --config finetune_essays/config.yaml
python finetune_essays/run.py --resume
python finetune_essays/run.py --rec-ckpt outputs/<other>/best     # start from another checkpoint
```

Key knobs (`config.yaml`): `data_root`, `base_expand_w/h` + `frame_jitter` + `margin_frac`
(frame handling), `aug_prob` (photometric strength), and the usual `trainer.*` (short run,
low LR — this is polishing on a small clean set). Uses `trainer_v2` (WSD + label smoothing).
