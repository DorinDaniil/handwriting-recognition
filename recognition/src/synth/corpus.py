"""Bilingual text sampling from folders of .txt: running-text lines + handwriting-error aug.

Folders are weighted per language (``*_text_weights``): a folder is picked by weight,
then a random file inside it. File paths are stored as one bytes blob + numpy offsets
(not millions of Python str objects) — small RAM and copy-on-write-safe across workers.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import CorpusConfig
from .rng import chance, choice, lerp, randint

logger = logging.getLogger(__name__)
_WS = re.compile(r"\s+")

_BUILTIN_RU = ("и в не на что быть он как это она по но они мы из который то за свой весь год от "
               "так для если время школа учитель ученик урок задача ответ вопрос пример текст "
               "тетрадь сегодня погода хорошо большой новый первый природа друзья").split()
_BUILTIN_EN = ("the of and to in a is that it was for on are as with they at be this from or one "
               "school teacher student lesson question answer number letter english text page line "
               "today weather good first nature friends house city book").split()

# curated Latin tokens that realistically appear inside Russian handwriting
# (brands / terms / acronyms). Used for stage-2 code-switching.
_CODE_SWITCH = ("Wi-Fi IT PDF online email Python Google USB OK CEO URL Windows GPS SMS HR PR "
                "Internet Word Excel YouTube iPhone Android API app web ok").split()

_RU_CONF = {"о": "а", "а": "о", "е": "и", "и": "е", "я": "е", "э": "е", "ё": "е",
            "с": "з", "з": "с", "т": "д", "д": "т", "б": "п", "п": "б", "ж": "ш", "ш": "ж"}
_EN_CONF = {"a": "e", "e": "a", "i": "e", "o": "a", "u": "a", "s": "z", "c": "s", "k": "c"}


class _FileList:
    """Compact, COW-safe list of file paths: one bytes blob + line offsets."""
    __slots__ = ("_blob", "_s", "_e")

    def __init__(self, blob: bytes):
        arr = np.frombuffer(blob, dtype=np.uint8)
        seps = np.where(arr == 0x0A)[0]                 # '\n' positions
        self._blob = blob
        self._s = np.concatenate(([0], seps + 1)).astype(np.int64)
        self._e = np.concatenate((seps, [len(blob)])).astype(np.int64)

    def __len__(self):
        return len(self._s)

    def pick(self, rng) -> str:
        k = int(rng.integers(0, len(self._s)))
        return self._blob[self._s[k]:self._e[k]].decode("utf-8", "ignore")


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


def _discover_dir(d, glob: str, cache_dir=None):
    """Return a _FileList for one dir; cached to a per-dir manifest (walked once)."""
    if not Path(d).exists():
        logger.warning("corpus: missing dir %s", d)
        return None
    key = hashlib.md5((str(Path(d).resolve()) + glob).encode()).hexdigest()[:16]
    cache = Path(cache_dir or Path(tempfile.gettempdir()) / "synth_corpus")
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / f"{key}.txt"
    if manifest.exists():
        blob = manifest.read_bytes()
    else:
        blob = "\n".join(_walk(str(d), (glob.lstrip("*").lower(),))).encode("utf-8")
        try:
            manifest.write_bytes(blob)
        except Exception as e:
            logger.warning("corpus: manifest not written (%s): %s", manifest, e)
    if not blob:
        return None
    fl = _FileList(blob)
    logger.info("corpus: %d files in %s", len(fl), d)
    return fl


class TextSampler:
    def __init__(self, cfg: CorpusConfig, ru_charset: str, en_charset: str):
        self.cfg = cfg
        self._sets = {"ru": set(ru_charset), "en": set(en_charset)}
        self._groups, self._dir_w = {}, {}
        for lang, dirs, weights in (("ru", cfg.ru_text_dirs, cfg.ru_text_weights),
                                    ("en", cfg.en_text_dirs, cfg.en_text_weights)):
            self._groups[lang], self._dir_w[lang] = self._build_groups(dirs, weights, cfg)
        self._builtin = {"ru": [w for w in _BUILTIN_RU if all(c in self._sets["ru"] for c in w)],
                         "en": [w for w in _BUILTIN_EN if all(c in self._sets["en"] for c in w)]}
        self._conf = {"ru": _RU_CONF, "en": _EN_CONF}
        logger.info("TextSampler: ru=%d en=%d files", self.n_files("ru"), self.n_files("en"))

    def _build_groups(self, dirs, weights, cfg):
        use_w = bool(weights) and len(weights) == len(dirs)
        if weights and not use_w:
            logger.warning("corpus: %d weights != %d dirs -> ignoring weights", len(weights), len(dirs))
        groups, ws = [], []
        for i, d in enumerate(dirs):
            fl = _discover_dir(d, cfg.glob, cfg.cache_dir)
            if fl and len(fl):
                groups.append(fl)
                ws.append(float(weights[i]) if use_w else float(len(fl)))
        return groups, ws

    def n_files(self, lang: str) -> int:
        return sum(len(fl) for fl in self._groups[lang])

    def sample(self, rng, t: float = 1.0):
        lang = "ru" if chance(rng, self.cfg.p_ru) else "en"
        cs = self._sets[lang]
        n = self._target_len(rng, t)
        s = self._real_line(rng, n, lang) if self._groups[lang] else self._builtin_line(rng, n, lang)
        s = _clean(s, cs) or self._builtin_line(rng, n, lang)
        if lang == "ru" and self.cfg.p_code_switch > 0 and chance(rng, self.cfg.p_code_switch):
            s = self._code_switch(s, rng)
            cs = cs | self._sets["en"]            # let the inserted Latin survive cleaning
        s = self._apply_errors(s, lang, rng)
        s = _clean(s, cs)
        if self.cfg.lowercase_prob and chance(rng, self.cfg.lowercase_prob):
            s = s.lower()
        return s, lang

    def _target_len(self, rng, t):
        lo, hi = self.cfg.len_chars
        return randint(rng, (lo, max(lo, int(lerp(lo, hi, t)))))

    def _real_line(self, rng, n, lang):
        fl = choice(rng, self._groups[lang], self._dir_w[lang])   # pick folder by weight
        raw = _read_flat(fl.pick(rng))                            # random file inside it
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

    def _builtin_line(self, rng, n, lang):
        words = self._builtin[lang]
        out, total = [], 0
        while total < n and words:
            w = words[int(rng.integers(0, len(words)))]
            out.append(w); total += len(w) + 1
        j = " ".join(out)
        return j[:n].rsplit(" ", 1)[0] if len(j) > n else j

    def _code_switch(self, s, rng):
        """Insert whole EN word(s) between RU words (never split a word)."""
        words = [w for w in s.split(" ") if w]
        if not words:
            return s
        k = randint(rng, (1, max(1, self.cfg.code_switch_max_tokens)))
        for _ in range(k):
            tok = ""
            if self._groups["en"] and chance(rng, self.cfg.p_code_switch_corpus):
                tok = self._en_token(rng)
            if not tok:
                tok = _CODE_SWITCH[int(rng.integers(0, len(_CODE_SWITCH)))]
            tok = _clean(tok, self._sets["en"])
            if not tok:
                continue
            pos = int(rng.integers(1, len(words) + 1)) if len(words) > 1 else len(words)
            words.insert(pos, tok)
        return " ".join(words)

    def _en_token(self, rng):
        """One or two whole words sampled from the EN corpus."""
        raw = _read_flat(choice(rng, self._groups["en"], self._dir_w["en"]).pick(rng))
        parts = [w for w in raw.split(" ") if w]
        if not parts:
            return ""
        i = int(rng.integers(0, len(parts)))
        n = 1 if rng.random() < 0.7 else 2
        return " ".join(parts[i:i + n])

    def _apply_errors(self, s, lang, rng):
        c = self.cfg
        if not s or not chance(rng, c.p_text_error):
            return s
        conf = self._conf[lang]
        chars = list(s)
        for i, ch in enumerate(chars):
            low = ch.lower()
            if low in conf and chance(rng, c.p_letter_sub):
                r = conf[low]
                chars[i] = r.upper() if ch.isupper() else r
        s = "".join(chars)
        if chance(rng, c.p_drop_punct):
            idx = [i for i, ch in enumerate(s) if not ch.isalnum() and not ch.isspace()]
            if idx:
                j = idx[int(rng.integers(0, len(idx)))]
                s = s[:j] + s[j + 1:]
        if chance(rng, c.p_typo):
            letters = [i for i, ch in enumerate(s) if ch.isalpha()]
            if letters:
                j = letters[int(rng.integers(0, len(letters)))]
                s = (s[:j] + s[j] + s[j:]) if rng.random() < 0.5 else (s[:j] + s[j + 1:])
        return _WS.sub(" ", s).strip()
