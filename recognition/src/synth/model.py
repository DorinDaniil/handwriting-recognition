"""TrOCR-small (EN+RU): load the English checkpoint and extend the vocab for Russian.

Loads the full pretrained model, then resize_token_embeddings to the (extended) tokenizer:
English embedding rows are kept, new Russian-token rows are added fresh. transformers is
imported lazily.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_PRETRAINED = "microsoft/trocr-small-handwritten"


@dataclass
class LoadReport:
    old_vocab: int
    new_vocab: int
    added: int

    def summary(self) -> str:
        if self.new_vocab == self.old_vocab:
            return f"loaded full pretrained weights (vocab {self.new_vocab})"
        return (f"loaded pretrained encoder + decoder layers; vocab {self.old_vocab} -> "
                f"{self.new_vocab}, decoder embeddings re-initialised")


def _special_ids(tokenizer):
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    return bos, eos, pad


def build_trocr_small(tokenizer, pretrained: str = DEFAULT_PRETRAINED, max_length: int = 128):
    from transformers import VisionEncoderDecoderModel

    import torch

    model = VisionEncoderDecoderModel.from_pretrained(pretrained)
    old_vocab = model.decoder.config.vocab_size
    new_vocab = len(tokenizer)
    if new_vocab != old_vocab:
        model.decoder.resize_token_embeddings(new_vocab)
        # a freshly-trained vocab does not align with the English ids -> reinit the
        # decoder embeddings/head (encoder + decoder transformer layers stay pretrained)
        std = getattr(model.decoder.config, "init_std", 0.02)
        emb = model.decoder.get_input_embeddings()
        torch.nn.init.normal_(emb.weight, mean=0.0, std=std)
        head = model.decoder.get_output_embeddings()
        if head is not None and head.weight is not emb.weight:
            torch.nn.init.normal_(head.weight, mean=0.0, std=std)
    report = LoadReport(old_vocab, new_vocab, new_vocab - old_vocab)

    bos, eos, pad = _special_ids(tokenizer)
    for cfg in (model.config, model.decoder.config):
        cfg.vocab_size = new_vocab
        cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id = bos, eos, pad
    model.config.decoder_start_token_id = bos
    model.config.pad_token_id, model.config.eos_token_id = pad, eos

    gc = model.generation_config
    gc.decoder_start_token_id, gc.eos_token_id, gc.pad_token_id = bos, eos, pad
    gc.max_length = max_length
    logger.info("TrOCR-small: %s", report.summary())
    return model, report


def build_processor(tokenizer, pretrained: str = DEFAULT_PRETRAINED):
    from transformers import AutoImageProcessor, TrOCRProcessor
    return TrOCRProcessor(image_processor=AutoImageProcessor.from_pretrained(pretrained),
                          tokenizer=tokenizer)
