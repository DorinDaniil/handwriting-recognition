# recognition_v2

Line recogniser for Russian + English handwriting, trained from scratch. The line detector
(`detection/`) is used as it is; the synthetic generator is reused from `recognition/src/synth`.

## What is different from `recognition/`

| | recognition (TrOCR-small) | recognition_v2 |
|---|---|---|
| Input | any line squeezed into 384×384 | height 96, width follows the aspect ratio (up to 1536) |
| Encoder | ViT on 16×16 patches, 576 tokens per line | conv stem + 12 transformer layers, one frame per 8 px of width |
| Output | BPE, 65k tokens | 152 characters, shared by both heads |
| Heads | autoregressive decoder | CTC head + autoregressive decoder, trained together |
| Parameters | 62.1M (33M of them vocabulary tables) | 35.3M |
| Training crops | tight dataset crops | page-like context; detector boxes where pages are available |

- **Resolution.** A 50-character line used to get 6–9 px per character; now a character is
  about 20 px wide, so stroke counts in "ш / и / м" are visible.
- **CTC head.** Ties every character to the place where it was written. It is a second
  training signal for the encoder and a fast way to read without a decoding loop.
- **One crop function.** `htr/geometry.py` grows a line box by a frame measured in box
  heights and warps it upright. Training, evaluation and inference all call it, and the
  inference frame is saved inside the checkpoint.
- **Blank crops.** Empty paper and show-through of the other side are trained with an empty
  label, so the recogniser stays silent on false detector boxes.

## Run

```bash
pip install -r requirements.txt
python scripts/smoke_test.py                              # one minute: the whole training path on toy lines

python scripts/train.py --config configs/pretrain.yaml    # synthetic pretraining
python scripts/detect_pages.py --labels /mnt/DATA2/dorin/handwriting-recognition/data/kaliningrad_essays/labels.txt
python scripts/train.py --config configs/finetune.yaml    # real data, starts from outputs/pretrain/best.pt

python scripts/evaluate.py --checkpoint outputs/finetune/best.pt          # per-set line metrics, both heads
python scripts/eval_pages.py --data /mnt/DATA2/dorin/handwriting-recognition/data/kaliningrad_essays \
    --rec-ckpt outputs/finetune/best.pt                                   # detector + recogniser, page CER / WER
```

`--resume` continues a run from `last.pt`. Every run writes `best.pt` (weights averaged over
recent steps, self-contained for inference), `last.pt`, `history.jsonl` and `config.yaml`.

`detect_pages.py` runs the detector over the labelled essay pages and stores which box is which
labelled line. With that file the `essays` source crops training lines from the detector's own
boxes; without it, from the labelled polygons.

`eval_pages.py` leaves out the pages the recogniser was trained on and writes into the same
`eval_metrics_kaliningrad_essays.json` as the other systems. `--frame X Y` tries other crop
margins without retraining, `--head ctc|decoder|auto` picks the reading mode.

## Inference

```python
from htr.geometry import crop_line, paper_colour
from htr.predictor import Recognizer

reader = Recognizer("outputs/finetune/best.pt", device="cpu", head="auto")
crops = [crop_line(page, box, reader.frame, paper_colour(page)) for box in detector_boxes]   # page: PIL image
texts = reader.recognize(crops)
```

`head="ctc"` reads in one pass, `"decoder"` decodes character by character, `"auto"` uses the
decoder only for lines where CTC is unsure.

## Data

Sources are listed in the config (`data.sources`) with a `weight` — the share of training
samples. `weight: 0` means evaluation only.

| kind | data | format |
|---|---|---|
| `tsv` | Cyrillic, School Notebooks RU / EN, CVL, IMGUR5K | crops + `path<TAB>text` |
| `pairs` | HWR200 lines | folder of png + folder of txt |
| `hf` | IAM | HuggingFace dataset |
| `pages` | essays | page images + `labels.txt` with line polygons |
| `synth` | generator from `recognition/src/synth` | text corpora + fonts |
| `blank` | empty paper, show-through | made from the generator's paper and ink |

Splits are the same as in `recognition/`: the existing test parts stay test parts. For essays
`test_frac` is 0.3, a superset of the old 10% test pages.

Labels are normalised before training (`htr/text.py`): one spelling for quotes and dashes that
look the same on paper, characters outside the alphabet dropped, lines with an annotator's
`[...]` mark skipped.

## Layout

```
configs/      pretrain.yaml, finetune.yaml
htr/
  text.py       alphabet and label normalisation
  geometry.py   line boxes: frame, upright crop, matching detector boxes to labelled lines
  image.py      fixed height, free width, batch padding
  augment.py    page-like context for pre-cropped lines, photometric noise
  sources.py    the data sources above
  batching.py   width-sorted batches under a pixel budget
  model.py      stem, encoder, CTC head, decoder
  trainer.py    training loop, evaluation, checkpoints
  predictor.py  Recognizer for inference
scripts/      train, evaluate, detect_pages, eval_pages, smoke_test
```
