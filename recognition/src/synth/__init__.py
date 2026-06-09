"""On-the-fly synthetic handwritten line generator (EN + RU)."""
from .backgrounds import PaperBackground
from .config import (
    EN_CHARSET,
    RU_CHARSET,
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
    "HandwrittenLineGenerator", "SynthConfig", "build_synth_cfg", "RU_CHARSET", "EN_CHARSET",
    "CorpusConfig", "FontConfig", "RenderConfig", "PaperConfig", "EffectsConfig", "OutputConfig",
    "TextSampler", "FontBank", "FontEntry", "LineRenderer", "PaperBackground", "Compositor",
    "EffectsPipeline", "fit_to_square", "resize_to_min_side", "make_generator",
]
