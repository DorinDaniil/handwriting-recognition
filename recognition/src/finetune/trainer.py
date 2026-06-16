from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .metrics import collect_predictions, compute_metrics

_AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


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


def _save(path, model, opt, sched, scaler, epoch, best):
    torch.save({"epoch": epoch, "best": best, "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict()}, path)


def train(model, processor, train_loader, val_loaders, cfg, device, resume=False):
    from transformers import get_cosine_schedule_with_warmup

    t = cfg.trainer
    model.to(device)
    out = Path(t.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(train_loader)
    total_steps = t.num_epochs * steps_per_epoch
    select = t.get("select_metric", "nes_char")
    print(f"steps/epoch {steps_per_epoch} | epochs {t.num_epochs} | total steps {total_steps} | "
          f"select by harmonic {select}")

    amp_dtype = _AMP[t.get("amp_dtype", "bf16")]
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.get("weight_decay", 0.01))
    scheduler = get_cosine_schedule_with_warmup(optimizer, t.warmup_steps, total_steps)

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
        for i, batch in enumerate(train_loader, 1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model(pixel_values=pixel_values, labels=labels).loss

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
                                **{name: m.to_dict() for name, m in metrics.items()}},
                               ensure_ascii=False) + "\n")
        _save(out / "last.pt", model, optimizer, scheduler, scaler, epoch + 1, best)
        if score > best:
            best = score
            model.save_pretrained(out / "best"); processor.save_pretrained(out / "best")
            print(f"  saved best ({select} harmonic {best:.4f})")
    print("done.")