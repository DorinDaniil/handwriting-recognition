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
    ru_text_weights: tuple[float, ...] = ()  # per-folder sampling weights (len == dirs); empty -> by file count
    en_text_weights: tuple[float, ...] = ()
    glob: str = "*.txt"
    cache_dir: str | None = None             # file-list manifest cache (delete to rescan)
    p_ru: float = 0.5                        # share of Russian lines
    len_chars: tuple[int, int] = (8, 50)
    p_hyphenate: float = 0.15                # end a line mid-word with '-'
    lowercase_prob: float = 0.0
    # text-error block (imitates handwriting mistakes; the error goes into the label too)
    p_text_error: float = 0.25               # chance a line gets any errors
    p_letter_sub: float = 0.06               # per-letter chance to swap a look/sound-alike (о->а, e->a)
    p_drop_punct: float = 0.35               # chance to drop one punctuation mark
    p_typo: float = 0.12                     # chance to double or drop one letter
    # code-switch block (stage-2): insert whole EN word(s) into RU lines. The EN word
    # is real text and goes into the label. 0 -> off (stage-1 untouched, draws no RNG).
    p_code_switch: float = 0.0               # chance a RU line gets an EN insertion
    p_code_switch_corpus: float = 0.3        # share of insertions drawn from the EN corpus vs curated list
    code_switch_max_tokens: int = 3          # up to this many whole EN words inserted


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
    p_pencil: float = 0.20
    baseline_wobble_px: tuple[float, float] = (0.0, 2.0)
    slant_deg: tuple[float, float] = (-6.0, 9.0)           # italic shear
    line_rotate_deg: tuple[float, float] = (-2.5, 2.5)     # whole-line tilt
    per_glyph_rot_deg: tuple[float, float] = (-3.0, 3.0)
    spacing_jitter: tuple[float, float] = (-0.05, 0.16)    # less negative -> words don't glue
    size_jitter: tuple[float, float] = (0.95, 1.06)
    ink_alpha: tuple[float, float] = (0.82, 1.0)
    pencil_alpha: tuple[float, float] = (0.55, 0.82)
    stroke_grain: float = 0.14
    pad_px: int = 6
    space_min_frac: float = 0.33             # min word gap as fraction of glyph height
    # curved baseline (stage-2): low-frequency arc + sine waviness laid down at glyph
    # placement (the paper stays straight). 0 -> off (stage-1 untouched, draws no RNG).
    p_curved_baseline: float = 0.0
    baseline_arc_px: tuple[float, float] = (4.0, 14.0)     # parabolic sag/bulge amplitude
    baseline_wave_px: tuple[float, float] = (2.0, 6.0)     # sine waviness amplitude
    baseline_wave_harmonics: int = 2


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
    vignette: tuple[float, float] = (0.0, 0.10)
    real_paper_dir: str | None = None
    use_cache_pool: bool = False


@dataclass
class EffectsConfig:
    p_show_through: float = 0.06
    ink_bleed_px: tuple[float, float] = (0.0, 0.5)
    p_elastic: float = 0.18
    elastic_alpha: tuple[float, float] = (8.0, 20.0)
    elastic_sigma: tuple[float, float] = (4.0, 6.0)
    p_grid_distort: float = 0.12
    p_perspective: float = 0.15
    perspective_scale: tuple[float, float] = (0.02, 0.045)
    p_baseline_curve: float = 0.10
    p_blur: float = 0.22
    p_motion_blur: float = 0.05
    p_gauss_noise: float = 0.20
    p_iso_noise: float = 0.08
    p_brightness_contrast: float = 0.35
    p_gamma: float = 0.15
    p_illumination: float = 0.18
    p_jpeg: float = 0.30
    jpeg_quality: tuple[int, int] = (55, 95)
    p_downscale: float = 0.15
    downscale_range: tuple[float, float] = (0.65, 0.92)
    # shadows (stage-2). 0 -> off (stage-1 untouched, draws no RNG).
    p_drop_shadow: float = 0.0               # blurred dark copy of the ink, slightly offset
    shadow_blur_px: tuple[float, float] = (1.0, 3.0)
    shadow_alpha: tuple[float, float] = (0.25, 0.55)
    shadow_offset_px: tuple[int, int] = (1, 4)
    p_cast_shadow: float = 0.0               # large soft cast shadow (hand / page fold)
    cast_shadow_strength: tuple[float, float] = (0.25, 0.55)


@dataclass
class NeighborConfig:
    """Stage-2: bleed of adjacent lines into the top/bottom edge — as when a line
    detector crops loosely and a sliver of the neighbour line creeps in. The neighbour
    text is a DISTRACTOR: it never enters the label, the model must learn to ignore it.
    0 -> off (stage-1 untouched, draws no RNG). Give it vertical room via output.margin_frac."""
    p_neighbor: float = 0.0                  # chance a line gets a neighbour sliver
    p_both_sides: float = 0.4                # given a neighbour, chance of both top and bottom
    visible_frac: tuple[float, float] = (0.12, 0.40)   # fraction of the neighbour line height shown
    max_chars: int = 40                      # cap neighbour length (it is only a sliver anyway)


@dataclass
class OutputConfig:
    min_side: int = 384                      # shorter side after resize; aspect kept
    max_side: int | None = None
    margin_frac: tuple[float, float] = (0.10, 0.28)   # paper around text (room for geometry, no text crop)
    min_height_px: int = 24


@dataclass
class SynthConfig:
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    font: FontConfig = field(default_factory=FontConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    effects: EffectsConfig = field(default_factory=EffectsConfig)
    neighbors: NeighborConfig = field(default_factory=NeighborConfig)
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
        effects=_sub(EffectsConfig, "effects"), neighbors=_sub(NeighborConfig, "neighbors"),
        output=_sub(OutputConfig, "output"),
        ru_charset=g("ru_charset", RU_CHARSET), en_charset=g("en_charset", EN_CHARSET),
        seed=int(g("seed", 42)), curriculum=bool(g("curriculum", True)),
        warmup_steps=int(g("warmup_steps", 4000)),
    )
