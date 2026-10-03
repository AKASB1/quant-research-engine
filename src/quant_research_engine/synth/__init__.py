"""Synthetic market generator with planted, known predictability."""

from quant_research_engine.synth.config import SynthConfig, load_synth_config
from quant_research_engine.synth.generate import Truth, close_to_close_returns, generate

__all__ = ["SynthConfig", "Truth", "close_to_close_returns", "generate", "load_synth_config"]
