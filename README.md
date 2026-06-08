# Handwriting Recognition — двухэтапный пайплайн HTR (русский)

Монорепозиторий из двух самостоятельных проектов, образующих пайплайн распознавания
рукописного текста:

```
страница рукописи
   │
   ▼  ┌─────────────────────────────┐
      │  detection/  (DBNet++)       │   находит строки текста (bounding-боксы)
      └─────────────────────────────┘
   │  кроп каждой строки
   ▼  ┌─────────────────────────────┐
      │  recognition/ (TrOCR ru)     │   распознаёт текст в строке
      └─────────────────────────────┘
   ▼
   распознанный текст
```

## [`detection/`](detection) — детекция строк (DBNet++)

Готовый проект: реимплементация [DBNet++](https://arxiv.org/abs/2202.10304) для построчной
детекции на рукописных страницах (backbone ResNet/ConvNeXt + DCNv2, FPN+ASF, DB-head,
AMP/EMA/cosine). Подробности и запуск — в [detection/README.md](detection/README.md).

```bash
cd detection
pip install -r requirements.txt
python train.py            # config.yaml
```

## [`recognition/`](recognition) — распознавание (TrOCR, русский)

Новый проект: дообучение TrOCR под русский рукописный почерк. Ядро — генератор
**синтетических рукописных строк на лету** (`src/synth`): шрифты + фоны тетрадей
(линейка/клетка) + аугментации, чтобы учиться без ручной построчной разметки.
Подробности, архитектура и дорожная карта — в [recognition/README.md](recognition/README.md).

```bash
cd recognition
pip install -r requirements.txt
python scripts/fetch_fonts.py
python scripts/demo_synth.py
```

---

Проекты независимы (свои `requirements.txt`, свои `src/`); запускаются каждый из своей папки.
Общий источник данных — HWR200. `data/`, `outputs/`, `labels/`, скачанные шрифты — в `.gitignore`.
