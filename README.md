# handwriting-recognition

An HTR pipeline of two independent projects: line detection, then text recognition.

## detection/
DBNet++, line detection on a page. See `detection/README.md`.

```
cd detection
pip install -r requirements.txt
python train.py
```

## recognition/
Fine-tuning TrOCR for Russian handwriting.
(`src/synth`). See `recognition/README.md`.

```
cd recognition
pip install -r requirements.txt
python scripts/fetch_fonts.py
python scripts/demo_synth.py
```

Each project runs from its own folder (its own `requirements.txt`, `src/`).
`data/`, `outputs/`, `labels/` and fonts are in `.gitignore`.
