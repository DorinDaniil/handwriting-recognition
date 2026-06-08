"""On-the-fly synthetic handwritten Russian (Cyrillic) text-line generator.

Quick start::

    from src.synth import HandwrittenLineGenerator, SynthConfig, make_generator

    gen = HandwrittenLineGenerator(SynthConfig())   # needs fonts in assets/fonts/
    rng = make_generator(base_seed=42, worker_id=0, draw_index=0)
    image, text = gen.sample(rng, step=10_000)      # (PIL.Image 384x384, "строка текста")

See ``scripts/demo_synth.py`` for a preview grid and ``notebooks/test_synth.ipynb``
for a curriculum sweep. The torch ``IterableDataset`` / TrOCR training wiring is
the next step (see the project README roadmap).
"""
from .backgrounds import PaperBackground
from .config import (
    DEFAULT_CHARSET,
    CorpusConfig,
    EffectsConfig,
    FontConfig,
    OutputConfig,
    PaperConfig,
    RenderConfig,
    SynthConfig,
    build_synth_cfg,
)
from .corpus import TextSampler
from .effects import Compositor, EffectsPipeline
from .fonts import FontBank, FontEntry
from .generator import HandwrittenLineGenerator, fit_to_square, resize_to_min_side
from .render import LineRenderer
from .rng import make_generator

__all__ = [
    "HandwrittenLineGenerator",
    "SynthConfig",
    "build_synth_cfg",
    "DEFAULT_CHARSET",
    "CorpusConfig",
    "FontConfig",
    "RenderConfig",
    "PaperConfig",
    "EffectsConfig",
    "OutputConfig",
    "TextSampler",
    "FontBank",
    "FontEntry",
    "LineRenderer",
    "PaperBackground",
    "Compositor",
    "EffectsPipeline",
    "fit_to_square",
    "resize_to_min_side",
    "make_generator",
]
