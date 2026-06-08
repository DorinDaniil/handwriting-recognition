# handwriting-recognition

Пайплайн HTR из двух независимых проектов: детекция строк, затем распознавание текста.

## detection/
DBNet++, детекция строк на странице. См. `detection/README.md`.
```
cd detection
pip install -r requirements.txt
python train.py
```

## recognition/
Дообучение TrOCR под русский рукописный. Ядро — генератор синтетических строк
(`src/synth`). См. `recognition/README.md`.
```
cd recognition
pip install -r requirements.txt
python scripts/fetch_fonts.py
python scripts/demo_synth.py
```

Каждый проект запускается из своей папки (свои `requirements.txt`, `src/`).
`data/`, `outputs/`, `labels/` и шрифты в `.gitignore`.
