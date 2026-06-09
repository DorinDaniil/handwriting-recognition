"""Config dataclasses for the bilingual (EN+RU) synthetic line generator."""
from __future__ import annotations

from dataclasses import dataclass, field

_DIGITS = "0123456789"
_PUNCT = " .,!?;:-—()\"'/%"

RU_CHARSET: str = (
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ" + _DIGITS + _PUNCT + "«»№"
)
EN_CHARSET: str = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + _DIGITS + _PUNCT + "&@#"
)

RGB = tuple[int, int, int]


@dataclass
class CorpusConfig:
    ru_text_dirs: tuple[str, ...] = ()
    en_text_dirs: tuple[str, ...] = ()
    glob: str = "*.txt"
    cache_dir: str | None = None             # file-list manifest cache (delete to rescan)
    p_ru: float = 0.5                        # share of Russian lines
    p_real: float = 0.75                     # mode mix: real text / word salad / random glyphs
    p_words: float = 0.10
    p_random: float = 0.15
    len_chars: tuple[int, int] = (8, 50)
    p_hyphenate: float = 0.15                # end a line mid-word with '-'
    flatten_newlines: bool = True
    p_digits_in_random: float = 0.35
    p_punct_in_random: float = 0.25
    lowercase_prob: float = 0.0


@dataclass
class FontConfig:
    ru_font_dirs: tuple[str, ...] = ("assets/fonts_ru",)
    en_font_dirs: tuple[str, ...] = ("assets/fonts_en",)
    sizes_px: tuple[int, int] = (28, 56)
    min_glyph_coverage: float = 0.80         # drops Latin-only fonts from the RU pool and vice-versa
    weight_by_coverage: bool = True


@dataclass
class RenderConfig:
    ink_colors: tuple[RGB, ...] = (
        (20, 20, 28), (28, 40, 120), (40, 60, 160), (70, 70, 75),
    )
    p_pencil: float = 0.25
    baseline_wobble_px: tuple[float, float] = (0.0, 2.5)
    slant_deg: tuple[float, float] = (-8.0, 12.0)          # italic shear
    line_rotate_deg: tuple[float, float] = (-3.0, 3.0)     # whole-line tilt
    per_glyph_rot_deg: tuple[float, float] = (-4.0, 4.0)
    spacing_jitter: tuple[float, float] = (-0.12, 0.20)
    size_jitter: tuple[float, float] = (0.92, 1.08)
    ink_alpha: tuple[float, float] = (0.80, 1.0)
    pencil_alpha: tuple[float, float] = (0.45, 0.78)
    stroke_grain: float = 0.20
    pad_px: int = 6


@dataclass
class PaperConfig:
    paper_colors: tuple[RGB, ...] = (
        (252, 250, 244), (245, 242, 230), (255, 255, 255), (238, 232, 210),
    )
    p_plain: float = 0.40
    p_ruled: float = 0.30
    p_grid: float = 0.22
    p_real_crop: float = 0.08
    rule_spacing_px: tuple[int, int] = (26, 40)
    grid_spacing_px: tuple[int, int] = (18, 30)
    rule_colors: tuple[RGB, ...] = ((150, 170, 210), (180, 180, 190))
    rule_alpha: tuple[float, float] = (0.25, 0.6)
    p_margin_line: float = 0.35
    margin_color: RGB = (200, 70, 70)
    fiber_noise: float = 0.015
    vignette: tuple[float, float] = (0.0, 0.12)
    real_paper_dir: str | None = None
    use_cache_pool: bool = False


@dataclass
class EffectsConfig:
    p_show_through: float = 0.10
    ink_bleed_px: tuple[float, float] = (0.0, 0.8)
    p_elastic: float = 0.30
    elastic_alpha: tuple[float, float] = (10.0, 40.0)
    elastic_sigma: tuple[float, float] = (4.0, 7.0)
    p_grid_distort: float = 0.20
    p_perspective: float = 0.25
    perspective_scale: tuple[float, float] = (0.02, 0.06)
    p_baseline_curve: float = 0.15
    p_blur: float = 0.30
    p_motion_blur: float = 0.10
    p_gauss_noise: float = 0.30
    p_iso_noise: float = 0.15
    p_brightness_contrast: float = 0.45
    p_gamma: float = 0.20
    p_illumination: float = 0.25
    p_jpeg: float = 0.35
    jpeg_quality: tuple[int, int] = (35, 92)
    p_downscale: float = 0.25
    downscale_range: tuple[float, float] = (0.5, 0.85)


@dataclass
class OutputConfig:
    min_side: int = 224                      # shorter side after resize; aspect kept
    max_side: int | None = None
    margin_frac: tuple[float, float] = (0.06, 0.22)
    min_height_px: int = 24


@dataclass
class SynthConfig:
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    font: FontConfig = field(default_factory=FontConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    effects: EffectsConfig = field(default_factory=EffectsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ru_charset: str = RU_CHARSET
    en_charset: str = EN_CHARSET
    seed: int = 42
    curriculum: bool = True
    warmup_steps: int = 4000

    def charset(self, lang: str) -> str:
        return self.ru_charset if lang == "ru" else self.en_charset


def _coerce(v):
    from collections.abc import Sequence
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
        return tuple(_coerce(x) for x in v)
    return v


def build_synth_cfg(node) -> SynthConfig:
    """Build a SynthConfig from a mapping / OmegaConf node (partial overrides ok)."""
    if node is None:
        return SynthConfig()

    def _sub(cls, key):
        raw = node.get(key) if hasattr(node, "get") else None
        if raw is None:
            return cls()
        return cls(**{f: _coerce(raw[f]) for f in cls.__dataclass_fields__ if f in raw})

    g = (lambda k, d: node.get(k, d)) if hasattr(node, "get") else (lambda k, d: d)
    return SynthConfig(
        corpus=_sub(CorpusConfig, "corpus"), font=_sub(FontConfig, "font"),
        render=_sub(RenderConfig, "render"), paper=_sub(PaperConfig, "paper"),
        effects=_sub(EffectsConfig, "effects"), output=_sub(OutputConfig, "output"),
        ru_charset=g("ru_charset", RU_CHARSET), en_charset=g("en_charset", EN_CHARSET),
        seed=int(g("seed", 42)), curriculum=bool(g("curriculum", True)),
        warmup_steps=int(g("warmup_steps", 4000)),
    )
