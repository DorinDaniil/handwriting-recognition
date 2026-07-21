"""Improved fine-tune trainer for TrOCR-small (drop-in for trainer.train).

Same signature and I/O as trainer.py, but with three hardcoded convergence upgrades
(no config changes needed) motivated by "best model arrives at epoch 26/30":

  1. WSD schedule (warmup -> stable plateau at peak LR -> short cosine decay to a 10% floor).
     Keeps a working LR almost the whole run instead of cosine-decaying to ~0 early, so the
     late epochs keep learning instead of stalling; the length of the plateau is robust to
     dataset size / enabled sources.
  2. Label smoothing 0.1 in the seq2seq loss (HF's default CE has none) — standard for TrOCR,
     helps generalization and stabilizes convergence.
  3. Decoupled weight decay: no decay on biases / LayerNorm params (decay only on 2D+ weights).

Everything else (peak lr, warmup_steps, weight_decay, epochs, AMP, grad clip, beam eval,
selection metric, checkpointing, resume) is unchanged and still read from the config.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .metrics import collect_predictions, compute_metrics

_AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# --- hardcoded strategy knobs -------------------------------------------------
LABEL_SMOOTHING = 0.1       # seq2seq loss smoothing
WSD_DECAY_FRAC = 0.2        # last 20% of steps are the decay ramp; the rest is warmup + plateau
WSD_MIN_LR_RATE = 0.1       # decay bottoms out at 10% of peak LR, not 0


def _quality(metrics, key):
    value = getattr(metrics, key)
    return 1.0 - value if key in ("cer", "wer") else value


def harmonic_mean(values):
    values = list(values)
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def evaluate(model, processor, val_loaders, device, num_beams, max_len):
    predictions = collect_predictions(model, processor, val_loaders, device, num_beams, max_len)
    return {name: compute_metrics(refs, preds) for name, (refs, preds) in predictions.items()}


def selection_score(metrics_by_lang, key):
    return harmonic_mean(_quality(m, key) for m in metrics_by_lang.values())


def _param_groups(model, weight_decay):
    """Two groups: decay (2D+ weights) and no-decay (biases, LayerNorm, 1D params)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def _wsd_lambda(warmup, total, decay_frac=WSD_DECAY_FRAC, floor=WSD_MIN_LR_RATE):
    decay_start = max(warmup + 1, int(total * (1.0 - decay_frac)))

    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / max(1, total - decay_start)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return fn


def _loss(model, pixel_values, labels):
    logits = model(pixel_values=pixel_values, labels=labels).logits
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=-100, label_smoothing=LABEL_SMOOTHING)


def _save(path, model, opt, sched, scaler, epoch, best):
    torch.save({"epoch": epoch, "best": best, "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict()}, path)


def train(model, processor, train_loader, val_loaders, cfg, device, resume=False):
    t = cfg.trainer
    model.to(device)
    out = Path(t.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(train_loader)
    total_steps = t.num_epochs * steps_per_epoch
    select = t.get("select_metric", "nes_char")
    print(f"[trainer_v2 | WSD + label-smooth {LABEL_SMOOTHING}] steps/epoch {steps_per_epoch} | "
          f"epochs {t.num_epochs} | total steps {total_steps} | select by harmonic {select}")

    amp_dtype = _AMP[t.get("amp_dtype", "bf16")]
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    optimizer = torch.optim.AdamW(_param_groups(model, t.get("weight_decay", 0.01)),
                                  lr=t.lr, weight_decay=t.get("weight_decay", 0.01))
    scheduler = LambdaLR(optimizer, _wsd_lambda(t.warmup_steps, total_steps))

    start_epoch, best = 0, -1.0
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"], ck["best"]
        print(f"resumed at epoch {start_epoch} (best {select} {best:.4f})")

    clip = t.get("grad_clip", 1.0)
    for epoch in range(start_epoch, t.num_epochs):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for i, batch in enumerate(train_loader, 1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = _loss(model, pixel_values, labels)
            running += loss.item(); seen += 1

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            scheduler.step()

            if i % t.get("log_every", 50) == 0:
                print(f"epoch {epoch + 1}/{t.num_epochs} step {i}/{steps_per_epoch} "
                      f"loss {loss.item():.3f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"{i / (time.time() - t0):.1f} it/s")

        metrics = evaluate(model, processor, val_loaders, device, t.get("num_beams", 1),
                           cfg.model.max_target_len)
        for name, m in metrics.items():
            print("  " + m.row(name))
        score = selection_score(metrics, select)
        print(f"  [epoch {epoch + 1}] harmonic {select} = {score:.4f}")

        with open(out / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch + 1, "select": select, "score": score,
                                "loss": running / max(seen, 1),
                                **{name: m.to_dict() for name, m in metrics.items()}},
                               ensure_ascii=False) + "\n")
        _save(out / "last.pt", model, optimizer, scheduler, scaler, epoch + 1, best)
        if score > best:
            best = score
            model.save_pretrained(out / "best"); processor.save_pretrained(out / "best")
            print(f"  saved best ({select} harmonic {best:.4f})")
    print("done.")
