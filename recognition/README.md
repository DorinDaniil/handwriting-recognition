# recognition

Распознавание рукописного текста (TrOCR, русский). Ядро — генератор синтетических
рукописных строк на лету (`src/synth`): шрифты + фоны (линейка/клетка) + аугментации.
Обучающая обвязка TrOCR — следующий шаг.

## Установка
```
pip install -r requirements.txt
python scripts/fetch_fonts.py        # кириллические рукописные шрифты -> assets/fonts/
```

## Запуск
```
python scripts/demo_synth.py         # превью assets/synth_preview.png + замер скорости
```
Ноутбуки: `notebooks/synth_usage.ipynb` (как пользоваться), `notebooks/test_synth.ipynb` (свип сложности).

## Использование
```python
import sys; sys.path.insert(0, ".")
from src.synth import HandwrittenLineGenerator, make_generator

gen = HandwrittenLineGenerator.from_dirs(text_dirs=["/path/txt1", "/path/txt2"],
                                         font_dirs="assets/fonts")
img, text = gen.sample(make_generator(42, 0, 0), step=10000)   # (PIL 384x384, "строка")
```
Тексты берутся из папок с `.txt` (`text_dirs`): случайный обход файлов, строки бегущего
текста разной длины, перенос на границе слова. Без `text_dirs` — встроенный словарь.
`render_line()` отдаёт строку натурального размера, `sample()` — letterbox 384x384 под TrOCR.

## Структура src/synth
```
config.py       SynthConfig + под-конфиги (диапазоны и вероятности)
rng.py          worker-safe RNG (make_generator) + curriculum (lerp/scale_p)
corpus.py       TextSampler: обход папок .txt, строки, перенос
fonts.py        FontBank: пул шрифтов, фильтр покрытия глифов, кэш ImageFont
render.py       LineRenderer: строка -> RGBA-чернила (наклон, дрожание, джиттер)
backgrounds.py  PaperBackground: цвет, линейка/клетка/поля, реальная бумага
effects.py      Compositor + EffectsPipeline (albumentations, иначе numpy/PIL)
generator.py    HandwrittenLineGenerator: оркестратор, sample/render_line/from_dirs/fit_to_square
assets.py       сканирование шрифтов, кэш покрытия, пул бумаги
```

## Конфиг
`SynthConfig` — вложенные dataclass. Часто меняемое: `corpus.text_dirs`, `corpus.len_chars`,
`corpus.p_hyphenate`, `font.font_dirs`, `paper.p_grid`/`p_ruled`, `output.keep_aspect`,
`warmup_steps`, `curriculum`. `build_synth_cfg(node)` собирает из OmegaConf.
`step` в `sample(rng, step)` управляет сложностью: 0 — простые строки, `warmup_steps` — полная.

## Реальные данные (для валидации, не синтетика)
- Cyrillic Handwriting Dataset (Kaggle): ~73k строк.
- HKR: https://github.com/abdoelsayed2016/HKR_Dataset
- стартовый чекпойнт: kazars24/trocr-base-handwritten-ru

## Дальше
torch-обвязка (`IterableDataset` + collator + метрики CER/WER) и `train_recognition.py`
(`Seq2SeqTrainer`, `max_steps`, валидация только на реальных строках).
