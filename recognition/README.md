# recognition

Bilingual (EN+RU) handwritten text recognition on TrOCR-small. TrOCR-small already knows
English; Russian is added by extending the vocabulary + fine-tuning. The core is a synthetic
handwritten-line generator (`src/synth`): fonts + backgrounds (ruled/grid) + augmentations,
with two text folders and two font folders per language.

## Install
```
pip install -r requirements.txt
```
Fonts live in `assets/fonts_ru` and `assets/fonts_en`. Bring your own from any folders:
```
python scripts/merge_fonts.py --src /my/fonts                    # auto RU/EN routing by coverage
python scripts/merge_fonts.py --ru-src /my/ru --en-src /my/en    # explicit per language
```
Or a free starter set: `python scripts/fetch_fonts.py`. You can also point the config
(`font.ru_font_dirs` / `en_font_dirs`) at your own folders without copying.

## Run
```
python scripts/demo_synth.py      # writes assets/synth_preview.png + prints throughput
```
Notebooks: `notebooks/synth_usage.ipynb`, `test_synth.ipynb`, `trocr_small.ipynb`.

## Generation (usage)
```python
import sys; sys.path.insert(0, ".")
from src.synth import HandwrittenLineGenerator, make_generator

gen = HandwrittenLineGenerator.from_dirs(
    ru_text_dirs=["/data/ru_texts"], en_text_dirs=["/data/en_texts"],
    ru_font_dirs="assets/fonts_ru", en_font_dirs="assets/fonts_en", p_ru=0.5)
img, text = gen.sample(make_generator(42, 0, 0), step=10000)   # (PIL RGB, short side 224)
```
Each line is monolingual (RU/EN by `p_ru`). Texts are sliced from `.txt` files in
`*_text_dirs` (varied length, `-` hyphenation); empty lists -> built-in word lists.
`p_words=p_random=0` -> real text only. `sample()` -> tight crop on paper
(`output.min_side`=224, no white margins); square for TrOCR -> `fit_to_square(img, 384)`.

## Tokenizer
Appending Russian tokens to the English byte-level BPE via `add_tokens` breaks space
reconstruction (RU decodes with spurious spaces inside words). So we train a **new byte-level
BPE on your EN+RU corpus** (`train_new_from_iterator`) — round-trip correct for both languages
and compact for Russian. Decoder embeddings are re-initialized (encoder and decoder layers
stay pretrained).
```
python scripts/train_tokenizer.py --ru-text-dirs /data/ru1 /data/ru2 --en-text-dirs /data/en --vocab-size 12000
# -> assets/tokenizer_bi ; prints a round-trip check at the end (must match)
```

## Pretraining on synthetic data
```
python scripts/run_pretrain.py --config configs/pretrain_small.yaml
python scripts/run_pretrain.py --resume
```
Phase 1: encoder frozen (`freeze_encoder_steps`) — lets the re-initialized embeddings/head
catch up; phase 2: everything unfrozen. Checkpoints: `outputs/.../best` (by CER) and `last.pt`
(resume).

## Fine-tuning
After pretraining, fine-tune on real RU+EN lines (lower LR, same loop):
```
python scripts/run_finetune.py --config configs/finetune.yaml
python scripts/run_finetune.py --resume
```
Data sources are listed in `configs/finetune.yaml` (`data.sources`, each with its own
`kind`/`lang`); downloading lives in `scripts/download/`. Evaluate a checkpoint per source:
```
python scripts/eval_finetune.py --checkpoint outputs/<run>/best   # CER/WER/NES_char/NES_word
```

## Layout
- `src/synth/` — synthetic generator (corpus/fonts/backgrounds/effects)
- `src/model.py` — `build_trocr_small` (vocab extension) + `build_processor`
- `src/data.py`, `src/finetune/` — datasets, augmentations, metrics
- `scripts/` — pretrain / finetune / eval / tokenizer / fonts / data download
- `configs/` — `pretrain_small.yaml`, `pretrain_stage2.yaml`, `finetune.yaml`
