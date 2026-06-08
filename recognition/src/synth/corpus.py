"""Target-string sampling.

Three modes, mixed per :class:`~synth.config.CorpusConfig`:
  * **real**   — a contiguous character span from a real corpus (HWR200 ``full_text``,
                 wiki, books). Matches the eval domain; best for low CER.
  * **words**  — a salad of real words; widens vocabulary coverage.
  * **random** — random glyphs incl. digits/punctuation; hardens rare characters.

Every returned string is guaranteed to be a subset of ``charset`` (so the label,
the glyphs, and the coverage check stay consistent). A small built-in word list
keeps the generator producing *readable* lines even before any corpus is wired
up — handy for the first visual sanity checks.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .config import CorpusConfig
from .rng import chance, choice, lerp, randint

logger = logging.getLogger(__name__)

# Minimal fallback vocabulary (common Russian words) so word/real modes work with
# zero configured corpora. Real training should point CorpusConfig at real text.
_BUILTIN_WORDS: tuple[str, ...] = (
    "и", "в", "не", "на", "что", "тот", "быть", "с", "он", "как", "это", "она",
    "по", "но", "они", "мы", "из", "у", "который", "то", "за", "свой", "весь",
    "год", "от", "так", "о", "для", "вы", "если", "время", "рука", "когда",
    "другой", "наш", "знать", "стать", "при", "человек", "жизнь", "день", "себя",
    "более", "город", "дом", "слово", "место", "лицо", "глаз", "вода", "земля",
    "работа", "книга", "школа", "учитель", "ученик", "урок", "задача", "ответ",
    "вопрос", "пример", "число", "буква", "язык", "русский", "текст", "письмо",
    "ручка", "тетрадь", "страница", "строка", "сегодня", "погода", "хорошо",
    "большой", "новый", "первый", "русская", "история", "природа", "друзья",
)


def _clean(s: str, charset_set: set[str]) -> str:
    """Drop out-of-charset characters (→ space), collapse runs of spaces, trim."""
    out = "".join(c if c in charset_set else " " for c in s)
    return re.sub(r"\s+", " ", out).strip()


class TextSampler:
    def __init__(self, cfg: CorpusConfig, charset: str):
        self.cfg = cfg
        self.charset_set = set(charset)
        alpha = [c for c in charset if c.isalpha()]
        # split case so random mode can prefer lowercase like real text
        self._lower = [c for c in alpha if c.islower()] or alpha
        self._upper = [c for c in alpha if c.isupper()]
        self._digits = [c for c in charset if c.isdigit()]
        self._punct = [c for c in charset if (not c.isalnum()) and (not c.isspace())]

        self._docs = self._load_docs(cfg.real_text_files)
        self._words = self._load_words(cfg.word_lists, self._docs)

        # renormalize available modes (drop real/words if we truly have nothing)
        modes, weights = [], []
        if self._docs:
            modes.append("real"); weights.append(cfg.p_real)
        if self._words:
            modes.append("words"); weights.append(cfg.p_words)
        modes.append("random"); weights.append(cfg.p_random if (self._docs or self._words) else 1.0)
        self._modes, self._mode_w = modes, weights

    # ----------------------------------------------------------- loading

    def _load_docs(self, files) -> list[str]:
        docs: list[str] = []
        for f in files:
            try:
                txt = Path(f).read_text(encoding="utf-8", errors="ignore")
                cleaned = _clean(txt, self.charset_set)
                if len(cleaned) > 20:
                    docs.append(cleaned)
            except Exception as e:
                logger.warning("corpus: could not read %s: %s", f, e)
        return docs

    def _load_words(self, files, docs) -> list[str]:
        words: set[str] = set(_BUILTIN_WORDS)
        for f in files:
            try:
                for line in Path(f).read_text(encoding="utf-8", errors="ignore").splitlines():
                    w = _clean(line, self.charset_set)
                    if w and " " not in w:
                        words.add(w)
            except Exception as e:
                logger.warning("corpus: could not read word list %s: %s", f, e)
        for d in docs:
            words.update(w for w in d.split(" ") if w)
        return [w for w in words if all(c in self.charset_set for c in w)]

    # ----------------------------------------------------------- sampling

    def _target_len(self, rng, t: float) -> int:
        lo, hi = self.cfg.len_chars
        hi_t = int(lerp(lo, hi, t))          # curriculum: shorter lines early
        return randint(rng, (lo, max(lo, hi_t)))

    def sample(self, rng, t: float = 1.0) -> str:
        mode = choice(rng, self._modes, self._mode_w)
        n = self._target_len(rng, t)
        if mode == "real":
            s = self._real_span(rng, n)
        elif mode == "words":
            s = self._word_salad(rng, n)
        else:
            s = self._random_glyphs(rng, n)
        s = _clean(s, self.charset_set)
        if not s:
            s = self._word_salad(rng, n) or self._random_glyphs(rng, n)
        if self.cfg.lowercase_prob and chance(rng, self.cfg.lowercase_prob):
            s = s.lower()
        return s

    def _real_span(self, rng, n: int) -> str:
        doc = self._docs[int(rng.integers(0, len(self._docs)))]
        if len(doc) <= n:
            return doc
        start = int(rng.integers(0, len(doc) - n))
        span = doc[start:start + n + 12]
        # snap to word boundaries to avoid cutting mid-word
        sp = span.find(" ")
        if 0 < sp < 6:
            span = span[sp + 1:]
        if len(span) > n:
            cut = span.rfind(" ", 0, n + 1)
            span = span[:cut] if cut > n // 2 else span[:n]
        return span.strip()

    def _word_salad(self, rng, n: int) -> str:
        out: list[str] = []
        total = 0
        while total < n:
            w = self._words[int(rng.integers(0, len(self._words)))]
            out.append(w)
            total += len(w) + 1
        return " ".join(out)[:n].rsplit(" ", 1)[0] if total > n else " ".join(out)

    def _random_glyphs(self, rng, n: int) -> str:
        chars: list[str] = []
        prev_space = True
        for _ in range(n):
            r = rng.random()
            if not prev_space and r < 0.16:
                chars.append(" "); prev_space = True; continue
            if self._digits and chance(rng, self.cfg.p_digits_in_random * 0.4):
                chars.append(self._digits[int(rng.integers(0, len(self._digits)))])
            elif self._punct and (not prev_space) and chance(rng, self.cfg.p_punct_in_random * 0.25):
                chars.append(self._punct[int(rng.integers(0, len(self._punct)))])
            elif self._upper and prev_space and chance(rng, 0.12):
                chars.append(self._upper[int(rng.integers(0, len(self._upper)))])
            else:
                chars.append(self._lower[int(rng.integers(0, len(self._lower)))])
            prev_space = chars[-1] == " "
        return "".join(chars).strip()
