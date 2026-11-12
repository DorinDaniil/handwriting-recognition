"""Line recogniser: conv stem -> transformer encoder -> CTC head + autoregressive decoder.

The input is an RGB line image of fixed height and free width. The stem turns it into one
feature vector per 8 px of width; both heads read that sequence and share one character
vocabulary. CTC ties every character to the place it was written, the decoder adds context.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text import BLANK, BOS, EOS

STRIDE = 8


@dataclass
class ModelConfig:
    vocab: int
    height: int = 96
    dim: int = 384
    heads: int = 6
    encoder_layers: int = 12
    decoder_layers: int = 4
    stem: tuple = (32, 64, 128, 192, 256)
    dropout: float = 0.1
    max_text_len: int = 160


def conv(c_in: int, c_out: int, stride=1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class Residual(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride=1):
        super().__init__()
        self.body = nn.Sequential(conv(c_in, c_out, stride),
                                  nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out))
        self.skip = None
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return F.relu(self.body(x) + (x if self.skip is None else self.skip(x)))


class Stem(nn.Module):
    """[B, 3, H, W] -> [B, W/8, dim]. Width is reduced 8x; height 32x, then folded away."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.stem
        self.layers = nn.Sequential(
            conv(3, c[0], 2), conv(c[0], c[0]),
            Residual(c[0], c[1], 2),
            Residual(c[1], c[2], 2), Residual(c[2], c[2]),
            Residual(c[2], c[3], (2, 1)), Residual(c[3], c[3]),
            Residual(c[3], c[4], (2, 1)), Residual(c[4], c[4]))
        self.fold = nn.Conv2d(c[4], cfg.dim, (cfg.height // 32, 1))

    def forward(self, images):
        return self.fold(self.layers(images)).squeeze(2).transpose(1, 2)


def _angles(positions, dim: int):
    rate = torch.exp(torch.arange(0, dim, 2, device=positions.device) * (-math.log(10000.0) / dim))
    return positions[:, None].float() * rate


def sinusoid(positions, dim: int):
    """Absolute position code [N, dim]: lets the decoder tell equal fragments of a line apart."""
    angle = _angles(positions, dim)
    return torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(1)


def rotate(x, positions):
    """Rotary position code: turn channel pairs of x [B, heads, N, d] by an angle set by position,
    so attention scores depend on the distance between tokens."""
    angle = _angles(positions, x.shape[-1])
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads, self.dropout = cfg.heads, cfg.dropout
        self.q, self.k, self.v = (nn.Linear(cfg.dim, cfg.dim) for _ in range(3))
        self.out = nn.Linear(cfg.dim, cfg.dim)

    def split(self, x):
        b, n, d = x.shape
        return x.view(b, n, self.heads, d // self.heads).transpose(1, 2)

    def attend(self, q, k, v, mask):
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).flatten(2))


class SelfAttention(Attention):
    """With a cache it reads one new token at a time and keeps the keys/values of the past ones."""

    def forward(self, x, mask=None, cache=None):
        start = cache["k"].shape[2] if cache else 0
        positions = torch.arange(start, start + x.shape[1], device=x.device)
        q = rotate(self.split(self.q(x)), positions)
        k = rotate(self.split(self.k(x)), positions)
        v = self.split(self.v(x))
        if cache is not None:
            if cache:
                k, v = torch.cat((cache["k"], k), 2), torch.cat((cache["v"], v), 2)
            cache["k"], cache["v"] = k, v
        return self.attend(q, k, v, mask)


class CrossAttention(Attention):
    """With a cache the encoder sequence is projected once and reused at every decoding step."""

    def forward(self, x, memory, mask, cache=None):
        if cache:
            k, v = cache["k"], cache["v"]
        else:
            k, v = self.split(self.k(memory)), self.split(self.v(memory))
            if cache is not None:
                cache["k"], cache["v"] = k, v
        return self.attend(self.split(self.q(x)), k, v, mask)


def mlp(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(cfg.dim), SelfAttention(cfg)
        self.norm2, self.mlp = nn.LayerNorm(cfg.dim), mlp(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.norm1(x), mask))
        return x + self.drop(self.mlp(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1, self.self_attn = nn.LayerNorm(cfg.dim), SelfAttention(cfg)
        self.norm2, self.cross_attn = nn.LayerNorm(cfg.dim), CrossAttention(cfg)
        self.norm3, self.mlp = nn.LayerNorm(cfg.dim), mlp(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, causal, memory_mask, cache=None):
        own, seen = (cache["self"], cache["cross"]) if cache is not None else (None, None)
        x = x + self.drop(self.self_attn(self.norm1(x), causal, own))
        x = x + self.drop(self.cross_attn(self.norm2(x), memory, memory_mask, seen))
        return x + self.drop(self.mlp(self.norm3(x)))


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.stem = Stem(cfg)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.encoder_layers))
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, images, widths):
        """-> features [B, N, dim], number of real (non-padding) frames per line, their mask [B, N]."""
        x = self.stem(images)
        n = x.shape[1]
        frames = torch.arange(n, device=x.device)
        lengths = torch.div(widths + STRIDE - 1, STRIDE, rounding_mode="floor").clamp(1, n)
        valid = frames[None] < lengths[:, None]
        x = x + sinusoid(frames, x.shape[-1]).to(x.dtype)
        for layer in self.layers:
            x = layer(x, valid[:, None, None, :])
        return self.norm(x), lengths, valid


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab, cfg.dim)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.decoder_layers))
        self.norm = nn.LayerNorm(cfg.dim)

    def new_cache(self) -> list[dict]:
        return [{"self": {}, "cross": {}} for _ in self.layers]

    def forward(self, tokens, memory, valid, cache=None):
        """Logits [B, T, vocab] for the next character after each token. With a cache, `tokens`
        holds only the newest token and the past is taken from the cache."""
        start = cache[0]["self"]["k"].shape[2] if cache and cache[0]["self"] else 0
        t, dim = tokens.shape[1], self.embed.embedding_dim
        positions = torch.arange(start, start + t, device=tokens.device)
        x = self.embed(tokens) * dim ** 0.5
        x = x + sinusoid(positions, dim).to(x.dtype)
        causal = None if cache is not None else torch.ones(t, t, dtype=torch.bool, device=tokens.device).tril()
        memory_mask = valid[:, None, None, :]
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, causal, memory_mask, None if cache is None else cache[i])
        return F.linear(self.norm(x), self.embed.weight)


def _init(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=module.embedding_dim ** -0.5)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")


class HTR(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.ctc = nn.Linear(cfg.dim, cfg.vocab)
        self.decoder = Decoder(cfg)
        self.apply(_init)

    def loss(self, batch, ctc_weight: float = 0.5, smoothing: float = 0.1):
        memory, lengths, valid = self.encoder(batch["images"], batch["widths"])
        frames = self.ctc(memory).float().log_softmax(-1).transpose(0, 1)
        ctc = F.ctc_loss(frames, batch["targets"], lengths, batch["target_lengths"],
                         blank=BLANK, zero_infinity=True)
        logits = self.decoder(batch["decoder_in"], memory, valid).float()
        ce = F.cross_entropy(logits.flatten(0, 1), batch["decoder_out"].flatten(),
                             ignore_index=-100, label_smoothing=smoothing)
        return ctc_weight * ctc + (1.0 - ctc_weight) * ce, {"ctc": ctc.item(), "ce": ce.item()}

    @torch.no_grad()
    def read(self, images, widths, head: str = "decoder"):
        """-> per line: character ids and a confidence in [0, 1]. head: "ctc" | "decoder"."""
        memory, lengths, valid = self.encoder(images, widths)
        return self.read_ctc(memory, lengths, valid) if head == "ctc" else self.read_decoder(memory, lengths, valid)

    def read_ctc(self, memory, lengths, valid):
        """Best path: most likely symbol per frame, repeats merged, blanks dropped."""
        prob, best = self.ctc(memory).float().softmax(-1).max(-1)
        confidence = (prob * valid).sum(1) / lengths
        lines = []
        for row, n, c in zip(best.tolist(), lengths.tolist(), confidence.tolist()):
            ids, previous = [], BLANK
            for token in row[:n]:
                if token != previous and token != BLANK:
                    ids.append(token)
                previous = token
            lines.append((ids, c))
        return lines

    def read_decoder(self, memory, lengths, valid):
        """Greedy character-by-character decoding of the whole batch with a key/value cache."""
        cache = self.decoder.new_cache()
        token = torch.full((memory.shape[0], 1), BOS, device=memory.device)
        done = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device)
        ids, scores = [], []
        for _ in range(min(self.cfg.max_text_len, int(lengths.max())) + 1):
            logits = self.decoder(token, memory, valid, cache)[:, -1].float()
            logits[:, [BLANK, BOS]] = -torch.inf
            score, token = logits.log_softmax(-1).max(-1, keepdim=True)
            ids.append(token)
            scores.append(score)
            done |= token[:, 0] == EOS
            if done.all():
                break
        lines = []
        for row, score in zip(torch.cat(ids, 1).tolist(), torch.cat(scores, 1).tolist()):
            n = row.index(EOS) if EOS in row else len(row)
            lines.append((row[:n], math.exp(sum(score[:n + 1]) / len(score[:n + 1]))))
        return lines
