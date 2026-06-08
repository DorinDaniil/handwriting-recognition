"""Target-string sampling — built for *folders of .txt files*.

Point :class:`~synth.config.CorpusConfig` at one or more ``text_dirs`` and the
sampler walks them at random: it discovers the ``.txt`` files once (paths only),
then per sample picks a random file, reads it lazily (LRU-cached), and cuts a
**running-text line** of varying length — optionally breaking the final word
with a hyphen (``перенос``), exactly as a line wraps in a real notebook. Lines of
different length naturally yield images of slightly different size.

Three modes (mixed via the ``p_*`` weights):
  * **real**   — a running-text span from your .txt corpus  (the main one);
  * **words**  — a salad of words (built-in list + optional ``word_lists``);
  * **random** — random glyphs incl. digits/punctuation (rare-char robustness).

Every returned string is a subset of ``charset`` so the label, the rendered
glyphs and the coverage check always agree.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from .config import CorpusConfig
from .rng import chance, choice, lerp, randint

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")

# Fallback vocabulary so word/real modes still produce readable lines with no corpus.
_BUILTIN_WORDS: tuple[str, ...] = (
    "и", "в", "не", "на", "что", "быть", "он", "как", "это", "она", "по", "но",
    "они", "мы", "из", "который", "то", "за", "свой", "весь", "год", "от", "так",
    "для", "если", "время", "рука", "когда", "другой", "наш", "знать", "стать",
    "человек", "жизнь", "день", "себя", "город", "дом", "слово", "место", "лицо",
    "вода", "земля", "работа", "книга", "школа", "учитель", "ученик", "урок",
    "задача", "ответ", "вопрос", "пример", "число", "буква", "язык", "русский",
    "текст", "письмо", "ручка", "тетрадь", "страница", "строка", "сегодня",
    "погода", "хорошо", "большой", "новый", "первый", "природа", "друзья",
)


@lru_cache(maxsize=48)
def _read_flat(path: str) -> str:
    """Decoded file as one running-text string (newlines/whitespace -> single spaces)."""
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:  # pragma: no cover
        logger.warning("corpus: cannot read %s: %s", path, e)
        return ""
    return _WS.sub(" ", txt).strip()


@lru_cache(maxsize=48)
def _read_lines(path: str) -> tuple[str, ...]:
    """Decoded file split into non-trivial source lines (kept as-is)."""
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ()
    return tuple(ln.strip() for ln in txt.splitlines() if len(ln.strip()) >= 2)


def _clean(s: str, charset_set: set[str]) -> str:
    out = "".join(c if c in charset_set else " " for c in s)
    return _WS.sub(" ", out).strip()


class TextSampler:
    def __init__(self, cfg: CorpusConfig, charset: str):
        self.cfg = cfg
        self.charset_set = set(charset)
        alpha = [c for c in charset if c.isalpha()]
        self._lower = [c for c in alpha if c.islower()] or alpha
        self._upper = [c for c in alpha if c.isupper()]
        self._digits = [c for c in charset if c.isdigit()]
        self._punct = [c for c in charset if (not c.isalnum()) and (not c.isspace())]

        self._files = self._discover(cfg)
        self._words = self._load_words(cfg.word_lists)

        modes, weights = [], []
        if self._files:
            modes.append("real"); weights.append(cfg.p_real)
        modes.append("words"); weights.append(cfg.p_words if self._files else max(cfg.p_words, 0.4))
        modes.append("random"); weights.append(cfg.p_random)
        self._modes, self._mode_w = modes, weights
        logger.info("TextSampler: %d corpus files, modes=%s", len(self._files), self._modes)

    # ----------------------------------------------------------- discovery

    def _discover(self, cfg: CorpusConfig) -> list[str]:
        files: list[str] = []
        seen: set[str] = set()
        for d in cfg.text_dirs:
            root = Path(d)
            if not root.exists():
                logger.warning("corpus: text_dir does not exist: %s", d)
                continue
            for p in root.rglob(cfg.glob):
                rp = str(p.resolve())
                if p.is_file() and rp not in seen:
                    seen.add(rp); files.append(rp)
        for f in cfg.real_text_files:
            rp = str(Path(f).resolve())
            if rp not in seen:
                seen.add(rp); files.append(rp)
        return files

    def _load_words(self, files) -> list[str]:
        words: set[str] = set(_BUILTIN_WORDS)
        for f in files:
            try:
                for line in Path(f).read_text(encoding="utf-8", errors="ignore").splitlines():
                    w = _clean(line, self.charset_set)
                    if w and " " not in w:
                        words.add(w)
            except Exception as e:
                logger.warning("corpus: cannot read word list %s: %s", f, e)
        return [w for w in words if w and all(c in self.charset_set for c in w)]

    # ----------------------------------------------------------- sampling

    def _target_len(self, rng, t: float) -> int:
        lo, hi = self.cfg.len_chars
        hi_t = int(lerp(lo, hi, t))               # curriculum: shorter lines early
        return randint(rng, (lo, max(lo, hi_t)))

    def sample(self, rng, t: float = 1.0) -> str:
        mode = choice(rng, self._modes, self._mode_w)
        n = self._target_len(rng, t)
        if mode == "real":
            s = self._real_line(rng, n)
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

    def _real_line(self, rng, n: int) -> str:
        """Random file -> running-text span of ~n chars, optionally hyphenated."""
        path = self._files[int(rng.integers(0, len(self._files)))]
        if not self.cfg.flatten_newlines:
            lines = _read_lines(path)
            if lines:
                ln = lines[int(rng.integers(0, len(lines)))]
                return self._cut(ln, rng, n, 0)
            return ""
        raw = _read_flat(path)
        if len(raw) <= 2:
            return ""
        if len(raw) <= n:
            return raw
        start = int(rng.integers(0, len(raw) - n))
        sp = raw.find(" ", start)                 # begin at a word boundary
        if 0 <= sp - start < 8:
            start = sp + 1
        return self._cut(raw, rng, n, start)

    def _cut(self, raw: str, rng, n: int, start: int) -> str:
        """Take raw[start:start+n], ending on a word boundary or a hyphen break."""
        end = min(start + n, len(raw))
        if end >= len(raw) or raw[end] == " " or raw[end - 1] == " ":
            return raw[start:end]
        last_space = raw.rfind(" ", start, end)
        if chance(rng, self.cfg.p_hyphenate):
            frag = end - (last_space + 1 if last_space != -1 else start)
            nxt = raw.find(" ", end)
            remain = (nxt if nxt != -1 else len(raw)) - end
            if frag >= 2 and remain >= 2:          # don't orphan 1 char on either side
                return raw[start:end] + "-"
        if last_space > start:                     # snap back to the last full word
            return raw[start:last_space]
        return raw[start:end]                      # single very long token: hard cut

    def _word_salad(self, rng, n: int) -> str:
        out, total = [], 0
        while total < n:
            w = self._words[int(rng.integers(0, len(self._words)))]
            out.append(w)
            total += len(w) + 1
        joined = " ".join(out)
        return joined[:n].rsplit(" ", 1)[0] if len(joined) > n else joined

    def _random_glyphs(self, rng, n: int) -> str:
        chars: list[str] = []
        prev_space = True
        for _ in range(n):
            if not prev_space and rng.random() < 0.16:
                chars.append(" ")
            elif self._digits and chance(rng, self.cfg.p_digits_in_random * 0.4):
                chars.append(self._digits[int(rng.integers(0, len(self._digits)))])
            elif self._punct and (not prev_space) and chance(rng, self.cfg.p_punct_in_random * 0.25):
                chars.append(self._punct[int(rng.integers(0, len(self._punct)))])
            elif self._upper and prev_space and chance(rng, 0.12):
                chars.append(self._upper[int(rng.integers(0, len(self._upper)))])
            else:
                chars.append(self._lower[int(rng.integers(0, len(self._lower)))])
            prev_space = chars[-1] == " "
        return "".join(chars).strip()
