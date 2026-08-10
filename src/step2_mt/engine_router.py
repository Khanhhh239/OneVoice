"""Direction-aware quality router backed by artifact-verified model choices."""
from __future__ import annotations

from .engine import NLLBEngine

MODEL_600M = "facebook/nllb-200-distilled-600M"
MODEL_1_3B = "facebook/nllb-200-1.3B"
FALLBACK_600M = {("en", "ko"), ("en", "zh")}


class DirectionalNLLBRouter:
    """Use 1.3B except where full FLORES showed a quality regression.

    Models are loaded lazily and cached. Keeping both resident needs roughly
    3.8 GiB of measured model memory, excluding framework overhead.
    """

    def __init__(self, device: str = "auto"):
        self.device = device
        self._engines: dict[str, NLLBEngine] = {}

    @staticmethod
    def model_for(src: str, tgt: str) -> str:
        return MODEL_600M if (src, tgt) in FALLBACK_600M else MODEL_1_3B

    def _engine(self, model: str) -> NLLBEngine:
        if model not in self._engines:
            self._engines[model] = NLLBEngine(model, self.device)
        return self._engines[model]

    def translate_batch(self, texts: list[str], src: str, tgt: str, **kwargs) -> list[str]:
        model = self.model_for(src, tgt)
        return self._engine(model).translate_batch(texts, src, tgt, **kwargs)

    def synchronize(self) -> None:
        for engine in self._engines.values():
            engine.synchronize()
