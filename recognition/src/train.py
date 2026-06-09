"""Synthetic pretrain loop for TrOCR-small (Russian).

Two-phase schedule (decided in lieu of a separate "tune" script):
  * phase 1 — the DeiT vision encoder is frozen for ``trainer.freeze_encoder_steps``
    so the freshly initialised decoder embeddings/head adapt without corrupting the
    pretrained visual features;
  * phase 2 — encoder unfrozen, everything trains end-to-end.

Infinite synthetic stream -> trained by ``max_steps``. Checkpoints: ``last.pt``
(resumable) and ``best/`` (HF ``save_pretrained``, by val CER). Logs to history.jsonl.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

_AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _set_encoder_trainable(model, flag: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad_(flag)


def _log(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_ckpt(path, model, opt, sched, scaler, step, best) -> None:
    torch.save({"step": step, "best": best, "model": model.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict()}, path)


def _load_ckpt(path, model, opt, sched, scaler, device):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
    return ck["step"], ck["best"]


@torch.no_grad()
def evaluate(model, processor, val_loader, device, num_beams, max_len):
    import jiwer
    model.eval()
    refs, preds = [], []
    for batch in val_loader:
        ids = model.generate(batch["pixel_values"].to(device),
                             num_beams=num_beams, max_length=max_len)
        preds += processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        refs += batch["texts"]
    model.train()
    return float(jiwer.cer(refs, preds)), float(jiwer.wer(refs, preds))


def train_model(model, processor, train_loader, val_loader, config,
                step_counter=None, resume: bool = False):
    from transformers import get_cosine_schedule_with_warmup

    t = config.trainer
    device = torch.device(config.get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    model.to(device)
    out = Path(t.output_dir); out.mkdir(parents=True, exist_ok=True)

    amp_dtype = _AMP[t.get("amp_dtype", "bf16")]
    use_amp = amp_dtype != torch.float32 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    frozen = t.get("freeze_encoder_steps", 0) > 0
    _set_encoder_trainable(model, not frozen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr,
                                  weight_decay=t.get("weight_decay", 0.01))
    scheduler = get_cosine_schedule_with_warmup(optimizer, t.warmup_steps, t.max_steps)

    step, best = 0, float("inf")
    if resume and (out / "last.pt").exists():
        step, best = _load_ckpt(out / "last.pt", model, optimizer, scheduler, scaler, device)
        frozen = frozen and step < t.freeze_encoder_steps
        _set_encoder_trainable(model, not frozen)
        print(f"resumed at step {step} (best CER {best:.4f})")

    model.train()
    it = iter(train_loader)
    t0, clip = time.time(), t.get("grad_clip", 1.0)
    while step < t.max_steps:
        if frozen and step >= t.freeze_encoder_steps:
            _set_encoder_trainable(model, True); frozen = False
            print(f"[step {step}] encoder unfrozen")

        batch = next(it)
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
        step += 1
        if step_counter is not None:
            step_counter.value = step

        if step % t.get("log_every", 50) == 0:
            its = step / (time.time() - t0)
            print(f"step {step}/{t.max_steps}  loss {loss.item():.3f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}  {its:.1f} it/s")

        if step % t.eval_every == 0 or step == t.max_steps:
            cer, wer = evaluate(model, processor, val_loader, device,
                                t.get("num_beams", 1), config.model.max_target_len)
            print(f"  [eval] step {step}  CER {cer:.4f}  WER {wer:.4f}")
            _log(out / "history.jsonl", {"step": step, "cer": cer, "wer": wer, "loss": loss.item()})
            _save_ckpt(out / "last.pt", model, optimizer, scheduler, scaler, step, best)
            if cer < best:
                best = cer
                model.save_pretrained(out / "best"); processor.save_pretrained(out / "best")
                print(f"  saved best (CER {best:.4f}) -> {out / 'best'}")
    print("done.")
