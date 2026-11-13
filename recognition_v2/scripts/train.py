#!/usr/bin/env python3
"""Train the recogniser. The config decides whether it is pretraining or fine-tuning.

    python scripts/train.py --config configs/pretrain.yaml
    python scripts/train.py --config configs/finetune.yaml
    python scripts/train.py --config configs/finetune.yaml --resume
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from htr.batching import train_loader
from htr.model import HTR, ModelConfig
from htr.sources import build_sources
from htr.text import Alphabet
from htr.trainer import train


def build_model(cfg):
    """A new model from the config, or the `init` checkpoint with the architecture it was saved with."""
    if not cfg.get("init"):
        alphabet = Alphabet()
        spec = OmegaConf.to_container(cfg.model)
        spec["stem"] = tuple(spec["stem"])
        return HTR(ModelConfig(vocab=len(alphabet), height=cfg.image.height, **spec)), alphabet
    saved = torch.load(ROOT / cfg.init, map_location="cpu")
    model = HTR(ModelConfig(**saved["config"]))
    model.load_state_dict(saved["model"])
    cfg.image = saved["image"]
    cfg.model = {"max_text_len": saved["config"]["max_text_len"]}
    print(f"initialised from {cfg.init}")
    return model, Alphabet(saved["alphabet"])


def main(config, resume, fresh):
    cfg = OmegaConf.load(config)
    cfg.run = str(ROOT / cfg.run)
    if (Path(cfg.run) / "last.pt").exists() and not (resume or fresh):
        raise SystemExit(f"{cfg.run} already holds a run: pass --resume to continue it "
                         f"or --fresh to start over and overwrite its checkpoints")
    torch.manual_seed(cfg.get("seed", 42))

    model, alphabet = build_model(cfg)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
          f"{len(alphabet.chars)} characters")
    sources = build_sources(cfg, alphabet, ROOT)
    counter = mp.Value("i", 0)
    loader = train_loader(sources, alphabet, cfg, counter)

    Path(cfg.run).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(cfg.run) / "config.yaml")
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    train(model, alphabet, loader, sources, cfg, device, counter, resume)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--resume", action="store_true", help="continue from last.pt of the run")
    ap.add_argument("--fresh", action="store_true", help="start over even if the run already has checkpoints")
    args = ap.parse_args()
    main(args.config, args.resume, args.fresh)
