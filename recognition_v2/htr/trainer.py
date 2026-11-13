"""Step-based training of the recogniser: one loop for synthetic pretraining and for fine-tuning.

Evaluation runs on an exponential moving average of the weights, with both heads, on every
held-out set; the best checkpoint is chosen by the harmonic mean of a metric over languages.
"""
from __future__ import annotations

import copy
import json
import math
import os
import signal
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from .batching import test_batches
from .metrics import harmonic, row, score

HEADS = ("ctc", "decoder")


class Ema:
    """Exponential moving average of the weights; the decay ramps up over the first steps."""

    def __init__(self, model, decay: float):
        self.decay, self.updates = decay, 0
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.state[k].lerp_(v, 1.0 - decay)
            else:
                self.state[k].copy_(v)


def _schedule(warmup: int, total: int, floor: float):
    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return rate


def _optimizer(model, cfg):
    decay, plain = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else plain).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": plain, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.98))


@torch.no_grad()
def read_set(model, alphabet, items, cfg, device, limit: int = 0):
    """-> references and, per head, hypotheses for one held-out set (one encoder pass for both)."""
    refs, hyps = [], {head: [] for head in HEADS}
    for images, widths, texts in test_batches(items, cfg, limit):
        memory, lengths, valid = model.encoder(images.to(device), widths.to(device))
        hyps["ctc"] += [alphabet.decode(ids) for ids, _ in model.read_ctc(memory, lengths, valid)]
        hyps["decoder"] += [alphabet.decode(ids) for ids, _ in model.read_decoder(memory, lengths, valid)]
        refs += texts
    return refs, hyps


def evaluate(model, alphabet, sources, cfg, device, limit: int = 0) -> dict:
    """-> {"sets": {name: {head: metrics}}, "langs": {lang: {head: metrics}}}."""
    model.eval()
    sets, pooled = {}, defaultdict(lambda: ([], {head: [] for head in HEADS}))
    for source in sources:
        if source.test is None:
            continue
        refs, hyps = read_set(model, alphabet, source.test, cfg, device, limit)
        sets[source.name] = {head: score(refs, hyps[head]) for head in HEADS}
        pooled[source.lang][0].extend(refs)
        for head in HEADS:
            pooled[source.lang][1][head].extend(hyps[head])
    langs = {lang: {head: score(refs, hyps[head]) for head in HEADS} for lang, (refs, hyps) in pooled.items()}
    return {"sets": sets, "langs": langs}


def report(results: dict):
    for group in ("sets", "langs"):
        for name, heads in results[group].items():
            for head in HEADS:
                print("  " + row(f"{name} [{head}]", heads[head]), flush=True)


def selection(results: dict, metric: str, head: str) -> float | None:
    """Harmonic mean of the metric over languages; None when there is nothing held out."""
    def quality(m):
        return 1.0 - m[metric] if metric in ("cer", "wer") else m[metric]
    if not results["langs"]:
        return None
    return harmonic(quality(heads[head]) for heads in results["langs"].values())


def save_weights(path, state, model, alphabet, cfg, **extra):
    """A self-contained checkpoint for inference: weights + everything needed to rebuild the model."""
    torch.save({"model": state, "config": asdict(model.cfg), "alphabet": alphabet.chars,
                "image": {"height": cfg.image.height, "max_width": cfg.image.max_width},
                "frame": list(cfg.frame.infer), **extra}, path)


def train(model, alphabet, loader, sources, cfg, device, counter, resume: bool = False):
    t = cfg.trainer
    out = Path(cfg.run)
    out.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = _optimizer(model, t)
    scheduler = LambdaLR(optimizer, _schedule(t.warmup_steps, t.max_steps, t.get("min_lr_ratio", 0.02)))
    ema = Ema(model, t.get("ema_decay", 0.9995))
    reader = copy.deepcopy(model)
    amp = device.type == "cuda" and t.get("amp", True)
    metric, head = t.get("select_metric", "nes_char"), t.get("select_head", "decoder")

    step, best = 0, -1.0
    if resume and (out / "last.pt").exists():
        last = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        ema.state, ema.updates = last["ema"], last["step"]
        step, best = last["step"], last["best"]
        print(f"resumed at step {step} (best {best:.4f})")
    counter.value = step

    print(f"steps {step} -> {t.max_steps} | eval every {t.eval_every} | "
          f"best by harmonic {metric} of the {head} head over languages")
    def save_state():
        torch.save({"model": model.state_dict(), "ema": ema.state, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "best": best}, out / "last.pt")

    trainer, stop = os.getpid(), []

    def on_interrupt(*_):
        """Ctrl+C: finish the current step, save and leave; a second Ctrl+C quits at once."""
        if os.getpid() != trainer:
            return
        if stop:
            raise KeyboardInterrupt
        stop.append(True)
        print("\nstopping after this step, the state will be saved (Ctrl+C again to quit at once)", flush=True)

    signal.signal(signal.SIGINT, on_interrupt)
    model.train()
    seen, lines, started = defaultdict(float), 0, time.time()
    for batch in loader:
        if step >= t.max_steps or stop:
            break
        texts = batch.pop("texts")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss, parts = model.loss(batch, cfg.loss.ctc_weight, cfg.loss.label_smoothing)
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, batch skipped")
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("grad_clip", 1.0))
        optimizer.step()
        scheduler.step()
        ema.update(model)
        step += 1
        counter.value = step

        lines += len(texts)
        for k, v in parts.items():
            seen[k] += v
        if step % t.log_every == 0:
            n = t.log_every
            print(f"step {step}/{t.max_steps} ctc {seen['ctc'] / n:.3f} ce {seen['ce'] / n:.3f} "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | {lines / (time.time() - started):.0f} lines/s "
                  f"| batch {len(texts)} x {batch['images'].shape[-1]} px", flush=True)
            seen, lines, started = defaultdict(float), 0, time.time()

        if step % t.eval_every == 0 or step == t.max_steps:
            reader.load_state_dict(ema.state)
            results = evaluate(reader, alphabet, sources, cfg, device, t.get("eval_limit", 0))
            report(results)
            value = selection(results, metric, head)
            with open(out / "history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "score": value, **results}, ensure_ascii=False) + "\n")
            if value is None or value > best:
                best = best if value is None else value
                save_weights(out / "best.pt", ema.state, model, alphabet, cfg, step=step, score=value)
                print(f"  [step {step}] saved best.pt" + ("" if value is None else f", harmonic {metric} {value:.4f}"))
            else:
                print(f"  [step {step}] harmonic {metric} {value:.4f} (best {best:.4f})")
            save_state()
            model.train()
            started, lines = time.time(), 0
    if stop:
        save_state()
        print(f"stopped at step {step}: state saved to last.pt, continue with --resume")
    else:
        print("done.")
