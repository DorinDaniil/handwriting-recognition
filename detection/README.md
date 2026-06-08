# Handwriting Text Detection — DBNet++

PyTorch reimplementation of [DBNet++ (arXiv:2202.10304)](https://arxiv.org/abs/2202.10304)
for line-level text detection on handwritten pages.

Backbones supported (via config): **ResNet-18 + DCNv2**, **ResNet-50 + DCNv2**, **ConvNeXt-Tiny**.

## Install

```bash
pip install -r requirements.txt
```

PyTorch ≥ 2.1 with CUDA recommended (training uses AMP / bf16).

## Data format

PaddleOCR `labels.txt`, one line per image:

```
<rel_path>\t[{"transcription": "...", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "score": 0.93}, ...]
```

Split files (`splits/train.txt`, `splits/val.txt`, `splits/test.txt`) are plain
text — one `rel_path` per line. Generated automatically on first training run if
absent, using `data.val_fraction` from the config.

## Train

```bash
# default config
python train.py

# custom config file
python train.py config.yaml

# CLI overrides (dotlist, OmegaConf-style)
python train.py trainer.epochs=200 data.batch_size=4 model.backbone.name=resnet50
```

Pick a backbone in `config.yaml`:

```yaml
model:
  backbone:
    name: resnet18        # | resnet50 | convnext_tiny
    pretrained: true
    use_dcn: true         # ignored for convnext_tiny
```

Outputs go to `outputs/<experiment.name>/`:
- `best.pt` — best validation H-mean
- `last.pt` — most recent
- `history.jsonl` — per-epoch metrics
- `config.resolved.yaml` — resolved config snapshot

Checkpoints carry both raw and EMA weights.

## Evaluate

```bash
python evaluate.py \
    --checkpoint outputs/dbnetpp_r18_hwr/best.pt \
    --dataset-root /path/to/images \
    --labels       labels/test_labels.txt \
    --split        splits/test.txt
```

Optional:
- `--iou-range 0.5,0.95,0.05` — COCO-style sweep (reports mean H-mean)
- `--pp-thresh / --pp-box-thresh / --pp-unclip / --pp-min-size` — tune
  post-processing without retraining
- `--no-ema` — evaluate raw model weights instead of EMA
- `--save-json out.json` — dump per-image predictions

Output: a single table with **Precision / Recall / H-mean / TP / FP / FN** per IoU.

## Notebooks

- `notebooks/test_augmentations.ipynb` — visualize augmentations + DBNet++ targets
- `notebooks/test_model_pipeline.ipynb` — smoke-test the model (shapes, gradients, FPS)
- `notebooks/inference.ipynb` — run a checkpoint on a single image, draw boxes
- `notebooks/filter_by_loss.ipynb` — clean noisy splits by per-sample loss

## Project layout

```
config.yaml              single training / inference config
train.py                 training entrypoint
evaluate.py              test-set metrics
src/
├── augmentation.py      polygon-safe augmentations (discrete rotations + flips)
├── dataset.py           PaddleOCR labels.txt -> torch Dataset
├── target_gen.py        DBNet++ target maps (prob / thresh + masks)
├── loss.py              OHEM BCE + Dice + masked L1
├── postprocess.py       prob map -> rotated quads (no cv2)
├── trainer.py           AMP, EMA, cosine LR + warmup, checkpoints
├── model/
│   ├── backbone.py      resnet18 / resnet34 / resnet50 / convnext_tiny (+ DCNv2)
│   ├── neck.py          FPN + ASF (Adaptive Scale Fusion)
│   ├── head.py          DBHead with differentiable binarization
│   └── dbnetpp.py       assembly + build_model factory
└── utils.py             seeding, drawing, preprocess_image_pil, H-mean
notebooks/               visual sanity checks + utilities
splits/                  train/val/test rel_path lists
labels/                  labels.txt (gitignored if large)
outputs/                 training runs (checkpoints, logs)
```
