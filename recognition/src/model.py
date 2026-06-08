"""TrOCR-small for Russian: the exact HF small architecture, English weights loaded partially.

The network is **not** re-implemented by hand. We instantiate the *exact* small
architecture from the pretrained config via HuggingFace (which is plain PyTorch —
``VisionEncoderDecoderModel`` is an ``nn.Module``), so the
``microsoft/trocr-small-handwritten`` weights fit by construction (identical config
=> identical tensor names/shapes).

For Russian we plug in a Cyrillic tokenizer. That changes only the decoder token
**embeddings** and the output **head** (their first dim is the vocab size); every
other tensor — the whole DeiT vision encoder and all decoder transformer layers —
is copied from the English checkpoint. The vision encoder is language-agnostic, so
this is the cheapest correct way to a Russian small model.

    from src.model import build_trocr_small, build_processor
    model, report = build_trocr_small(tokenizer)        # English small weights, RU embeddings
    print(report.summary())
    processor = build_processor(tokenizer)              # DeiT image preproc + your tokenizer

`transformers` is imported lazily (inside the functions), so importing this module
does not require it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_PRETRAINED = "microsoft/trocr-small-handwritten"


@dataclass
class LoadReport:
    loaded: int
    reinit: list = field(default_factory=list)        # model params left at fresh init
    skipped_src: list = field(default_factory=list)   # pretrained tensors that did not fit

    def summary(self) -> str:
        head = ", ".join(self.reinit[:6]) + (" ..." if len(self.reinit) > 6 else "")
        return (f"loaded {self.loaded} tensors from pretrained; "
                f"re-initialised {len(self.reinit)} (new-vocab embeddings/head): {head}")


def load_matching_state_dict(model: nn.Module, src_state: dict) -> LoadReport:
    """Copy tensors from ``src_state`` into ``model`` where name AND shape match.

    Vocab-sized tensors (decoder ``embed_tokens`` / output head) won't match a
    different tokenizer, so they stay at their fresh init — this is exactly the
    "load only part of the weights" step. Never raises on a mismatch; reports it."""
    tgt = model.state_dict()
    to_load = {k: v for k, v in src_state.items() if k in tgt and tgt[k].shape == v.shape}
    skipped = [k for k in src_state if k not in to_load]
    incompat = model.load_state_dict(to_load, strict=False)
    return LoadReport(loaded=len(to_load), reinit=list(incompat.missing_keys), skipped_src=skipped)


def _special_ids(tokenizer):
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    return bos, eos, pad


def build_trocr_small(tokenizer, pretrained: str = DEFAULT_PRETRAINED, max_length: int = 64):
    """Build TrOCR-small with the exact pretrained architecture, load matching weights.

    Returns ``(model, LoadReport)``. Pass the *English* tokenizer to load 100% of the
    weights (byte-level BPE already covers Cyrillic); pass a Russian tokenizer to get
    a Cyrillic vocab with only the embeddings/head re-initialised."""
    from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel

    config = VisionEncoderDecoderConfig.from_pretrained(pretrained)
    bos, eos, pad = _special_ids(tokenizer)

    # decoder vocab + special tokens -> the only structural change vs the English model
    config.decoder.vocab_size = len(tokenizer)
    config.decoder.bos_token_id = bos
    config.decoder.eos_token_id = eos
    config.decoder.pad_token_id = pad
    config.decoder_start_token_id = bos
    config.pad_token_id = pad
    config.eos_token_id = eos
    config.vocab_size = len(tokenizer)

    model = VisionEncoderDecoderModel(config=config)        # exact small arch, fresh weights

    pre = VisionEncoderDecoderModel.from_pretrained(pretrained)
    report = load_matching_state_dict(model, pre.state_dict())
    del pre

    gc = model.generation_config
    gc.decoder_start_token_id, gc.eos_token_id, gc.pad_token_id = bos, eos, pad
    gc.max_length = max_length
    logger.info("TrOCR-small RU: %s", report.summary())
    return model, report


def build_processor(tokenizer, pretrained: str = DEFAULT_PRETRAINED):
    """TrOCRProcessor = the pretrained DeiT image preprocessor (384, normalisation)
    paired with *your* tokenizer. Use ``processor(images=...).pixel_values`` and
    ``processor.tokenizer(text)`` (the image side is language-agnostic)."""
    from transformers import AutoImageProcessor, TrOCRProcessor
    image_processor = AutoImageProcessor.from_pretrained(pretrained)
    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
