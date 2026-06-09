"""Bilingual text sampling from folders of .txt (running-text lines, optional hyphenation)."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from .config import CorpusConfig
from .rng import chance, choice, lerp, randint

logger = logging.getLogger(__name__)
_WS = re.compile(r"\s+")

_BUILTIN_RU = (
    "и в не на что быть он как это она по но они мы из который то за свой весь год от так "
    "для если время рука когда другой наш знать стать человек жизнь день город дом слово "
    "место школа учитель ученик урок задача ответ вопрос пример число буква русский текст "
    "тетрадь страница строка сегодня погода хорошо большой новый первый природа друзья"
).split()
_BUILTIN_EN = (
    "the of and to in a is that it was for on are as with his they at be this from or one had "
    "by word but not what all were we when your can said there use each which she how their "
    "school teacher student lesson question answer number letter english text page line today "
    "weather good first nature friends house city book"
).split()


@lru_cache(maxsize=64)
def _read_flat(path: str) -> str:
    try:
        return _WS.sub(" ", Path(path).read_text(encoding="utf-8", errors="ignore")).strip()
    except Exception as e:
        logger.warning("corpus: cannot read %s: %s", path, e)
        return ""


def _clean(s: str, charset_set: set) -> str:
    return _WS.sub(" ", "".join(c if c in charset_set else " " for c in s)).strip()


def _walk(root: str, exts: tuple) -> list[str]:
    out, stack = [], [root]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.name.lower().endswith(exts):
                        out.append(e.path)
        except OSError:
            pass
    return out


def _discover(dirs, glob: str, cache_dir=None) -> list[str]:
    """List files under dirs; cached to a manifest so millions of files are walked once."""
    if not dirs:
        return []
    key = hashlib.md5(("|".join(sorted(map(str, dirs))) + glob).encode()).hexdigest()[:16]
    cache = Path(cache_dir or Path(tempfile.gettempdir()) / "synth_corpus")
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / f"{key}.txt"
    if manifest.exists():
        files = manifest.read_text(encoding="utf-8").splitlines()
        logger.info("corpus: %d files (cached)", len(files))
        return files
    exts = (glob.lstrip("*").lower(),)
    files = []
    for d in dirs:
        if Path(d).exists():
            files += _walk(str(d), exts)
        else:
            logger.warning("corpus: missing dir %s", d)
    manifest.write_text("\n".join(files), encoding="utf-8")
    logger.info("corpus: walked %d files (cached -> %s)", len(files), manifest)
    return files


class TextSampler:
    def __init__(self, cfg: CorpusConfig, ru_charset: str, en_charset: str):
        self.cfg = cfg
        self._sets = {"ru": set(ru_charset), "en": set(en_charset)}
        self._files = {"ru": _discover(cfg.ru_text_dirs, cfg.glob, cfg.cache_dir),
                       "en": _discover(cfg.en_text_dirs, cfg.glob, cfg.cache_dir)}
        self._words = {"ru": [w for w in _BUILTIN_RU if all(c in self._sets["ru"] for c in w)],
                       "en": [w for w in _BUILTIN_EN if all(c in self._sets["en"] for c in w)]}
        for lang, cset in (("ru", ru_charset), ("en", en_charset)):
            cs = self._sets[lang]
            up = [c for c in cset if c.isupper()]
            lo = [c for c in cset if c.islower()]
            dig = [c for c in cs if c.isdigit()]
            pun = [c for c in cs if not c.isalnum() and not c.isspace()]
            setattr(self, f"_{lang}_pools", (lo or up, up, dig, pun))
        logger.info("TextSampler: ru=%d en=%d files", len(self._files["ru"]), len(self._files["en"]))

    def sample(self, rng, t: float = 1.0):
        lang = "ru" if chance(rng, self.cfg.p_ru) else "en"
        cs = self._sets[lang]
        n = self._target_len(rng, t)
        mode = choice(rng, *self._modes(lang))
        if mode == "real":
            s = self._real_line(rng, n, lang)
        elif mode == "words":
            s = self._word_salad(rng, n, lang)
        else:
            s = self._random_glyphs(rng, n, lang)
        s = _clean(s, cs) or self._word_salad(rng, n, lang)
        if self.cfg.lowercase_prob and chance(rng, self.cfg.lowercase_prob):
            s = s.lower()
        return s, lang

    def _modes(self, lang):
        modes, w = [], []
        if self._files[lang]:
            modes.append("real"); w.append(self.cfg.p_real)
        modes.append("words"); w.append(self.cfg.p_words if self._files[lang] else max(self.cfg.p_words, 0.4))
        modes.append("random"); w.append(self.cfg.p_random)
        return modes, w

    def _target_len(self, rng, t):
        lo, hi = self.cfg.len_chars
        return randint(rng, (lo, max(lo, int(lerp(lo, hi, t)))))

    def _real_line(self, rng, n, lang):
        files = self._files[lang]
        raw = _read_flat(files[int(rng.integers(0, len(files)))])
        if len(raw) <= 2:
            return ""
        if len(raw) <= n:
            return raw
        start = int(rng.integers(0, len(raw) - n))
        sp = raw.find(" ", start)
        if 0 <= sp - start < 8:
            start = sp + 1
        return self._cut(raw, rng, n, start)

    def _cut(self, raw, rng, n, start):
        end = min(start + n, len(raw))
        if end >= len(raw) or raw[end] == " " or raw[end - 1] == " ":
            return raw[start:end]
        last_space = raw.rfind(" ", start, end)
        if chance(rng, self.cfg.p_hyphenate):
            frag = end - (last_space + 1 if last_space != -1 else start)
            nxt = raw.find(" ", end)
            remain = (nxt if nxt != -1 else len(raw)) - end
            if frag >= 2 and remain >= 2:
                return raw[start:end] + "-"
        return raw[start:last_space] if last_space > start else raw[start:end]

    def _word_salad(self, rng, n, lang):
        words = self._words[lang]
        out, total = [], 0
        while total < n and words:
            w = words[int(rng.integers(0, len(words)))]
            out.append(w); total += len(w) + 1
        joined = " ".join(out)
        return joined[:n].rsplit(" ", 1)[0] if len(joined) > n else joined

    def _random_glyphs(self, rng, n, lang):
        lo, up, dig, pun = getattr(self, f"_{lang}_pools")
        chars, prev_space = [], True
        for _ in range(n):
            if not prev_space and rng.random() < 0.16:
                chars.append(" ")
            elif dig and chance(rng, self.cfg.p_digits_in_random * 0.4):
                chars.append(dig[int(rng.integers(0, len(dig)))])
            elif pun and (not prev_space) and chance(rng, self.cfg.p_punct_in_random * 0.25):
                chars.append(pun[int(rng.integers(0, len(pun)))])
            elif up and prev_space and chance(rng, 0.12):
                chars.append(up[int(rng.integers(0, len(up)))])
            else:
                chars.append(lo[int(rng.integers(0, len(lo)))])
            prev_space = chars[-1] == " "
        return "".join(chars).strip()
