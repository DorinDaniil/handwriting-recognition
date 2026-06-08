"""Configuration for the synthetic handwritten-line generator.

Mirrors the dataclass-config style of the detection project
(see ``detection/src/augmentation.py``): every knob is a documented field with a
sensible default, ranges are ``(lo, hi)`` sampled uniformly, and ``p_*`` fields
are probabilities in ``[0, 1]``. Nested groups keep the eventual YAML readable
and let the training-time *curriculum* scale each group independently
(see :func:`synth.rng.scale_p` / :func:`synth.rng.lerp`).

Nothing here imports torch / albumentations — this module is pure data so it can
be unit-tested and built from an OmegaConf node exactly like the detection
configs are (``_build_aug_cfg`` in ``detection/train.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Full active character set. ё/Ё/й/ъ/щ are kept deliberately — they are the
# glyphs most often missing from free "handwriting" fonts, so they must be part
# of the coverage check (see synth.fonts.FontBank).
DEFAULT_CHARSET: str = (
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "0123456789"
    " .,!?;:-—()«»\"'/№%"
)

RGB = tuple[int, int, int]


@dataclass
class CorpusConfig:
    """Where target strings come from and how long they are."""
    real_text_files: tuple[str, ...] = ()   # .txt dumps: HWR200 full_text, wiki, books
    word_lists: tuple[str, ...] = ()         # frequency word lists for the word-salad mode
    # mode mix (renormalized internally) — coherent real text matches the eval domain,
    # word-salad widens vocabulary, random glyphs harden rare chars/digits/punct
    p_real: float = 0.55
    p_words: float = 0.30
    p_random: float = 0.15
    len_chars: tuple[int, int] = (8, 48)     # target line length, in characters
    p_digits_in_random: float = 0.35
    p_punct_in_random: float = 0.25
    allow_latin_mix: float = 0.05            # rare ru/en mixing seen in real notes
    lowercase_prob: float = 0.0              # TrOCR-handwritten is cased; keep 0 normally


@dataclass
class FontConfig:
    """The pool of Cyrillic handwriting fonts and how it is sampled."""
    font_dirs: tuple[str, ...] = ("assets/fonts",)
    sizes_px: tuple[int, int] = (28, 56)     # glyph pixel size range (pre-cached integers)
    # drop a font if it covers less than this fraction of the active charset; rare
    # punctuation (№ « » —) that a font lacks is dropped per-line by FontEntry.filter,
    # so this only needs to be high enough to reject Latin-only / uppercase-only fonts.
    min_glyph_coverage: float = 0.85
    weight_by_coverage: bool = True          # prefer fonts that cover more of the charset
    cache_warm: bool = True                  # pre-open (font, size) combos in worker_init_fn


@dataclass
class RenderConfig:
    """Per-glyph ink rendering: the handwriting "feel" lives here."""
    ink_colors: tuple[RGB, ...] = (
        (20, 20, 28),     # near-black ink
        (28, 40, 120),    # blue ballpoint
        (40, 60, 160),    # lighter blue
        (70, 70, 75),     # graphite / faded
    )
    p_pencil: float = 0.25                    # lower alpha + more grain (pencil look)
    baseline_wobble_px: tuple[float, float] = (0.0, 2.5)   # smooth vertical undulation amplitude
    slant_deg: tuple[float, float] = (-8.0, 12.0)          # line-level shear, mostly right-leaning
    per_glyph_rot_deg: tuple[float, float] = (-4.0, 4.0)
    spacing_jitter: tuple[float, float] = (-0.12, 0.20)    # fraction of glyph advance width
    size_jitter: tuple[float, float] = (0.92, 1.08)        # per-glyph scale
    ink_alpha: tuple[float, float] = (0.80, 1.0)
    pencil_alpha: tuple[float, float] = (0.45, 0.78)
    stroke_grain: float = 0.20               # dry-pen alpha texture strength (0 disables)
    pad_px: int = 6                          # transparent padding around the tight crop


@dataclass
class PaperConfig:
    """The substrate the ink is composited onto."""
    paper_colors: tuple[RGB, ...] = (
        (252, 250, 244),  # cream
        (245, 242, 230),  # aged
        (255, 255, 255),  # white
        (238, 232, 210),  # tan
    )
    # substrate type mix (renormalized) — клетка (grid) is very common in RU notebooks
    p_plain: float = 0.40
    p_ruled: float = 0.30                     # линейка (horizontal rules)
    p_grid: float = 0.22                      # клетка (square grid)
    p_real_crop: float = 0.08                 # paste onto a crop from a real-paper pool
    rule_spacing_px: tuple[int, int] = (26, 40)
    grid_spacing_px: tuple[int, int] = (18, 30)
    rule_colors: tuple[RGB, ...] = ((150, 170, 210), (180, 180, 190))
    rule_alpha: tuple[float, float] = (0.25, 0.6)
    p_margin_line: float = 0.35               # red vertical поля
    margin_color: RGB = (200, 70, 70)
    fiber_noise: float = 0.015               # paper grain (std of per-pixel gaussian)
    vignette: tuple[float, float] = (0.0, 0.12)
    real_paper_dir: str | None = None
    use_cache_pool: bool = False             # sample pre-rendered substrates (faster, see assets)


@dataclass
class EffectsConfig:
    """Compositing + capture degradation. Geometry is LINE-SAFE: no flips, no 90° rotations."""
    p_show_through: float = 0.10
    ink_bleed_px: tuple[float, float] = (0.0, 0.8)
    # geometry
    p_elastic: float = 0.30
    elastic_alpha: tuple[float, float] = (10.0, 40.0)
    elastic_sigma: tuple[float, float] = (4.0, 7.0)
    p_grid_distort: float = 0.20
    p_perspective: float = 0.25
    perspective_scale: tuple[float, float] = (0.02, 0.06)
    p_affine_rotate: float = 0.5
    affine_rotate_deg: tuple[float, float] = (-3.0, 3.0)
    p_baseline_curve: float = 0.15           # custom: sinusoidal row remap (albumentations has none)
    # photometric / capture
    p_blur: float = 0.30
    p_motion_blur: float = 0.10
    p_gauss_noise: float = 0.30
    p_iso_noise: float = 0.15
    p_brightness_contrast: float = 0.45
    p_gamma: float = 0.20
    p_illumination: float = 0.25             # custom: smooth lighting gradient multiply
    p_jpeg: float = 0.35
    jpeg_quality: tuple[int, int] = (35, 92)
    p_downscale: float = 0.25                # scan-style blur via downscale→upscale
    downscale_range: tuple[float, float] = (0.5, 0.85)


@dataclass
class OutputConfig:
    """How the line is delivered to the model."""
    proc_size: int = 384                      # TrOCR square input
    keep_aspect: bool = True                  # letterbox (pad), do NOT squash the aspect ratio
    pad_color: RGB = (255, 255, 255)
    max_aspect: float = 8.0                   # clamp extreme wide lines before letterboxing
    min_height_px: int = 24                   # reject degenerate renders below this height


@dataclass
class SynthConfig:
    """Top-level config aggregating every stage of the generator."""
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    font: FontConfig = field(default_factory=FontConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    effects: EffectsConfig = field(default_factory=EffectsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    charset: str = DEFAULT_CHARSET
    seed: int = 42
    curriculum: bool = True                   # ramp difficulty t: 0 -> 1 over warmup_steps
    warmup_steps: int = 4000


def build_synth_cfg(node) -> SynthConfig:
    """Build a :class:`SynthConfig` from a mapping / OmegaConf node.

    Mirrors the ``_build_*_cfg`` helpers in ``detection/train.py``: unknown keys
    are ignored and missing keys fall back to dataclass defaults, so a partial
    YAML override is enough. Pass ``None`` to get the full default config.
    """
    if node is None:
        return SynthConfig()

    def _sub(cls, key):
        raw = node.get(key) if hasattr(node, "get") else node[key] if key in node else None
        if raw is None:
            return cls()
        kwargs = {f: _coerce(raw[f]) for f in cls.__dataclass_fields__ if f in raw}
        return cls(**kwargs)

    return SynthConfig(
        corpus=_sub(CorpusConfig, "corpus"),
        font=_sub(FontConfig, "font"),
        render=_sub(RenderConfig, "render"),
        paper=_sub(PaperConfig, "paper"),
        effects=_sub(EffectsConfig, "effects"),
        output=_sub(OutputConfig, "output"),
        charset=node.get("charset", DEFAULT_CHARSET) if hasattr(node, "get") else DEFAULT_CHARSET,
        seed=int(node.get("seed", 42)) if hasattr(node, "get") else 42,
        curriculum=bool(node.get("curriculum", True)) if hasattr(node, "get") else True,
        warmup_steps=int(node.get("warmup_steps", 4000)) if hasattr(node, "get") else 4000,
    )


def _coerce(v):
    """OmegaConf returns ListConfig for sequences; convert to plain tuples so the
    dataclasses hold hashable, picklable values (safe to ship to DataLoader workers)."""
    try:
        from collections.abc import Sequence
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            return tuple(_coerce(x) for x in v)
    except Exception:
        pass
    return v
