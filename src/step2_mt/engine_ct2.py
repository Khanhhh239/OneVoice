"""CTranslate2 NLLB engine for CPU/GPU int8 deployment benchmarks."""
from __future__ import annotations

from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer

try:
    from .engine import LANG_CODES
except ImportError:
    from engine import LANG_CODES


class CT2NLLBEngine:
    def __init__(self, model_dir: str | Path, tokenizer_id: str = "facebook/nllb-200-distilled-600M",
                 device: str = "cpu", compute_type: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.translator = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type)

    def translate_batch(self, texts: list[str], src: str, tgt: str, *, num_beams: int = 1,
                        max_new_tokens: int = 256) -> list[str]:
        self.tokenizer.src_lang = LANG_CODES[src]
        token_batches = [self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text)) for text in texts]
        prefixes = [[LANG_CODES[tgt]]] * len(texts)
        results = self.translator.translate_batch(
            token_batches, target_prefix=prefixes, beam_size=num_beams,
            max_decoding_length=max_new_tokens)
        output = []
        for result in results:
            ids = self.tokenizer.convert_tokens_to_ids(result.hypotheses[0][1:])
            output.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return output

    def synchronize(self) -> None:
        return None
