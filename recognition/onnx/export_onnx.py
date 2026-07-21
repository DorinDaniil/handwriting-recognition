#!/usr/bin/env python3
"""Export the TrOCR checkpoint to ONNX for serving (run WHERE torch/optimum are installed).

Produces in --out:
  encoder_model.onnx            pixel_values -> encoder_hidden_states           (runs once)
  decoder_model.onnx            first step: input_ids + enc states -> logits + KV cache
  decoder_with_past_model.onnx  one generation step: last token + past KV -> logits + KV
  tokenizer.json                byte-level BPE, loadable by the light `tokenizers` lib
  service_config.json           image size / mean / std / special token ids / max_length

  python onnx/export_onnx.py --ckpt outputs/trocr_small_bi_finetune_with_hwr200_cleaned/best
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from optimum.exporters.onnx import main_export

ROOT = Path(__file__).resolve().parents[1]           # recognition/


OPSET = 18   # torchscript exporters support 18+ starting with torch 2.4


def export_graphs(ckpt: Path, out: Path):
    """optimum does the heavy lifting: encoder + decoder + decoder_with_past."""
    print(f"+ main_export({ckpt} -> {out}, task=image-to-text-with-past, opset={OPSET})")
    main_export(model_name_or_path=str(ckpt), output=str(out),
                task="image-to-text-with-past", opset=OPSET, no_post_process=True)
    for name in ("encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"):
        if not (out / name).exists():
            raise SystemExit(f"export incomplete: {name} missing in {out}")


def export_tokenizer(ckpt: Path, out: Path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    if (out / "tokenizer.json").exists():
        return tok
    if getattr(tok, "backend_tokenizer", None) is not None:      # fast tokenizer
        tok.backend_tokenizer.save(str(out / "tokenizer.json"))
    else:
        tok.save_pretrained(str(out))                             # should emit tokenizer.json
    if not (out / "tokenizer.json").exists():
        raise SystemExit("no tokenizer.json produced — service needs it (tokenizers lib)")
    return tok


def export_service_config(ckpt: Path, out: Path, tok):
    from transformers import AutoImageProcessor, GenerationConfig
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    gen = {}
    try:
        gen = GenerationConfig.from_pretrained(str(ckpt)).to_dict()
    except Exception:
        pass

    def pick(key, default=None):
        return gen.get(key) if gen.get(key) is not None else cfg.get(key, default)

    size, mean, std, resample = 384, [0.5] * 3, [0.5] * 3, 2
    try:
        ip = AutoImageProcessor.from_pretrained(str(ckpt))
        s = ip.size
        size = int(s["height"] if isinstance(s, dict) else s)
        mean, std = list(map(float, ip.image_mean)), list(map(float, ip.image_std))
        resample = int(getattr(ip, "resample", 2))
    except Exception as e:
        print(f"[warn] image processor not loadable ({e}); using TrOCR defaults 384/0.5/bilinear")

    svc = {
        "image_size": size, "image_mean": mean, "image_std": std, "resample": resample,
        "decoder_start_token_id": pick("decoder_start_token_id"),
        "eos_token_id": pick("eos_token_id"),
        "pad_token_id": pick("pad_token_id"),
        "vocab_size": len(tok),
        "max_length": int(pick("max_length", 128) or 128),
    }
    for k in ("decoder_start_token_id", "eos_token_id", "pad_token_id"):
        if svc[k] is None:
            raise SystemExit(f"{k} missing in config/generation_config — cannot serve")
    (out / "service_config.json").write_text(json.dumps(svc, indent=2), encoding="utf-8")
    print("service_config.json:", svc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "outputs/trocr_small_bi_finetune_with_hwr200_cleaned/best")
    ap.add_argument("--out", type=Path, default=ROOT / "onnx_out")
    args = ap.parse_args()

    ckpt = args.ckpt if args.ckpt.exists() else ROOT / args.ckpt
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    args.out.mkdir(parents=True, exist_ok=True)

    export_graphs(ckpt, args.out)
    tok = export_tokenizer(ckpt, args.out)
    export_service_config(ckpt, args.out, tok)

    # optimum also copies the whole HF config/tokenizer sidecar files — the service doesn't
    # need them (it runs on the 5 files below), so keep the folder clean
    keep = {"encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx",
            "tokenizer.json", "service_config.json"}
    removed = [p.name for p in args.out.iterdir() if p.is_file() and p.name not in keep]
    for name in removed:
        (args.out / name).unlink()
    if removed:
        print("removed HF sidecar files:", ", ".join(sorted(removed)))
    print(f"\ndone -> {args.out}\nservice needs: encoder/decoder/decoder_with_past .onnx + "
          f"tokenizer.json + service_config.json\ntry it with onnx/infer_onnx.ipynb")


if __name__ == "__main__":
    main()
