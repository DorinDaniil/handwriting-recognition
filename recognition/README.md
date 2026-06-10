# recognition

Двуязычное (EN+RU) распознавание рукописного текста на TrOCR-small. TrOCR-small уже
знает английский; русский добавляем расширением словаря + дообучением на синтетике.
Ядро — генератор синтетических рукописных строк (`src/synth`): шрифты + фоны
(линейка/клетка) + аугментации, по две папки текстов и шрифтов на язык.

## Установка
```
pip install -r requirements.txt
```
Шрифты лежат в `assets/fonts_ru` и `assets/fonts_en`. Перенести свои из любых папок:
```
python scripts/merge_fonts.py --src /мои/шрифты                   # авто-роутинг RU/EN по покрытию
python scripts/merge_fonts.py --ru-src /мои/ru --en-src /мои/en   # явно по языкам
```
Либо стартовый бесплатный набор: `python scripts/fetch_fonts.py`. Свои папки можно
указать и прямо в конфиге (`font.ru_font_dirs` / `en_font_dirs`) без копирования.

## Запуск
```
python scripts/demo_synth.py      # превью assets/synth_preview.png + скорость
```
Ноутбуки: `notebooks/synth_usage.ipynb`, `test_synth.ipynb`, `trocr_small.ipynb`.

## Генерация (использование)
```python
import sys; sys.path.insert(0, ".")
from src.synth import HandwrittenLineGenerator, make_generator

gen = HandwrittenLineGenerator.from_dirs(
    ru_text_dirs=["/data/ru_texts"], en_text_dirs=["/data/en_texts"],
    ru_font_dirs="assets/fonts_ru", en_font_dirs="assets/fonts_en", p_ru=0.5)
img, text = gen.sample(make_generator(42, 0, 0), step=10000)   # (PIL RGB, короткая сторона 224)
```
Каждая строка моноязычна (RU/EN по `p_ru`). Тексты режутся из `.txt` в `*_text_dirs`
(разная длина, перенос `-`); пустые списки -> встроенные словари. `p_words=p_random=0` ->
только реальный текст. `sample()` -> тугой кроп на бумаге (`output.min_side`=224, без белых
полей); квадрат под TrOCR -> `fit_to_square(img, 384)`.

## Токенайзер (двуязычный)
Дописывание русских токенов в английский byte-level BPE через `add_tokens` ломает
восстановление пробелов (RU декодится с лишними пробелами внутри слов). Поэтому обучаем
**новый byte-level BPE на твоём EN+RU корпусе** (`train_new_from_iterator`) — round-trip
корректный для обоих языков, русский компактный. Эмбеддинги декодера переинициализируются
(энкодер и слои декодера остаются претренированными).
```
python scripts/train_tokenizer.py --ru-text-dirs /data/ru1 /data/ru2 --en-text-dirs /data/en --vocab-size 12000
# -> assets/tokenizer_bi ; в конце печатает round-trip проверку (должно совпасть)
```
Без своего токенайзера (`model.tokenizer: null`) берётся английский byte-BPE: корректно, но RU ~2 токена/символ.

## Обучение (pretrain на синтетике)
```
python scripts/run_pretrain.py --config configs/pretrain_small.yaml
python scripts/run_pretrain.py --resume
```
Фаза 1: энкодер заморожен (`freeze_encoder_steps`) — догоняют переинициализированные эмбеддинги/голова;
фаза 2: всё размораживается. Чекпойнты `outputs/.../best` (по CER) и `last.pt` (resume).

## Структура
```
src/synth/      генератор: config, rng, corpus(EN/RU), fonts(EN/RU), render, backgrounds, effects, generator
src/model.py    build_trocr_small (расширение словаря) + build_processor
src/data.py     SynthLineDataset (бесконечный) + FixedSynthValDataset + TrOCRCollator
src/train.py    train_model: freeze->unfreeze, AMP, cosine+warmup, CER/WER, чекпойнты
scripts/        fetch_fonts, train_tokenizer, demo_synth, run_pretrain
configs/        pretrain_small.yaml
```

## Конфиг (часто меняемое)
`corpus.ru_text_dirs`/`en_text_dirs`, `corpus.p_ru`, `len_chars`, `p_hyphenate`,
`font.ru_font_dirs`/`en_font_dirs`, `paper.p_grid`/`p_ruled`, `output.min_side`,
`warmup_steps`, `curriculum`. `step` в `sample(rng, step)` управляет сложностью.

## Реальные данные (валидация / finetune, не синтетика)
- Cyrillic Handwriting Dataset (Kaggle, ~73k строк), HKR (github.com/abdoelsayed2016/HKR_Dataset).
- английские строки — IAM и т.п.
- Чекпойнты выбираем по реальному CER; синтетика — только обучение.

## Дальше
Finetune с `outputs/.../best` на реальных RU+EN строках (тот же цикл, реальный лоадер, ниже LR).
