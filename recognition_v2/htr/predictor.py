"""Inference: line crops in, text out.

    reader = Recognizer("outputs/finetune/best.pt", device="cpu", head="auto")
    texts = reader.recognize(crops)                 # crops made with reader.frame around detector boxes

head: "ctc"      one encoder pass, no decoding loop — fastest
      "decoder"  character-by-character decoding — most accurate
      "auto"     CTC for every line, the decoder only where CTC is unsure
"""
from __future__ import annotations

import torch
from PIL import Image

from .geometry import Frame
from .image import fit_height, to_batch
from .model import HTR, ModelConfig
from .text import Alphabet


class Recognizer:
    def __init__(self, checkpoint, device="cpu", head: str = "decoder", batch_size: int = 32,
                 threshold: float = 0.9):
        saved = torch.load(checkpoint, map_location="cpu")
        self.alphabet = Alphabet(saved["alphabet"])
        self.model = HTR(ModelConfig(**saved["config"]))
        self.model.load_state_dict(saved["model"])
        self.device = torch.device(device)
        self.model.eval().to(self.device)
        self.height, self.max_width = saved["image"]["height"], saved["image"]["max_width"]
        self.frame = Frame.uniform(*saved["frame"])
        self.head, self.batch_size, self.threshold = head, batch_size, threshold

    @torch.no_grad()
    def read(self, crops) -> list[tuple[str, float]]:
        """-> (text, confidence) per crop, in the order given."""
        arrays = [fit_height(c if isinstance(c, Image.Image) else Image.fromarray(c), self.height, self.max_width)
                  for c in crops]
        order = sorted(range(len(arrays)), key=lambda i: arrays[i].shape[1])
        out = [None] * len(arrays)
        for at in range(0, len(order), self.batch_size):
            index = order[at:at + self.batch_size]
            images, widths = to_batch([arrays[i] for i in index])
            memory, lengths, valid = self.model.encoder(images.to(self.device), widths.to(self.device))
            if self.head == "decoder":
                lines = self.model.read_decoder(memory, lengths, valid)
            else:
                lines = self.model.read_ctc(memory, lengths, valid)
                unsure = [j for j, (_, confidence) in enumerate(lines) if confidence < self.threshold]
                if self.head == "auto" and unsure:
                    for j, line in zip(unsure, self.model.read_decoder(memory[unsure], lengths[unsure], valid[unsure])):
                        lines[j] = line
            for i, (ids, confidence) in zip(index, lines):
                out[i] = (self.alphabet.decode(ids), confidence)
        return out

    def recognize(self, crops) -> list[str]:
        return [text for text, _ in self.read(crops)]
