"""Text normalisation and the character alphabet shared by the CTC head and the decoder."""
from __future__ import annotations

import re
import unicodedata

RU = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
EN = "abcdefghijklmnopqrstuvwxyz"
ALPHABET = RU + RU.upper() + EN + EN.upper() + "0123456789" + " .,!?;:-—()\"'/%«»№&@#+=*"

BLANK, BOS, EOS = 0, 1, 2
SPECIALS = 3

_SAME_GLYPH = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "‘": "'", "’": "'", "‚": "'", "`": "'", "´": "'", "′": "'",
    "‐": "-", "‑": "-", "‒": "-", "−": "-",
    "―": "—",
    "…": "...",
    " ": " ", " ": " ", " ": " ",
})
_SPACED_EN_DASH = re.compile(r"(?<=\s)–(?=\s)")
_SPACES = re.compile(r"\s+")
_UNREADABLE = re.compile(r"\[[^\]]*\]")


def normalize(text: str) -> str:
    """One spelling per glyph: quotes, dashes and spaces that look the same on paper."""
    text = unicodedata.normalize("NFC", text).translate(_SAME_GLYPH)
    text = _SPACED_EN_DASH.sub("—", text).replace("–", "-")
    return _SPACES.sub(" ", text).strip()


class Alphabet:
    def __init__(self, chars: str = ALPHABET):
        self.chars = chars
        self.index = {c: i + SPECIALS for i, c in enumerate(chars)}

    def __len__(self) -> int:
        return len(self.chars) + SPECIALS

    def clean(self, text: str) -> str | None:
        """Label as the model will see it, or None when it cannot be trusted: an annotator's
        [unreadable] mark, or too many characters outside the alphabet."""
        text = normalize(text)
        if _UNREADABLE.search(text):
            return None
        kept = "".join(c for c in text if c in self.index)
        if len(kept) < 0.9 * len(text):
            return None
        return _SPACES.sub(" ", kept).strip()

    def encode(self, text: str) -> list[int]:
        return [self.index[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.chars[i - SPECIALS] for i in ids if i >= SPECIALS)
