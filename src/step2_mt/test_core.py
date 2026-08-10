"""Fast model-free regression tests for packaging and CI."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from .benchmark import load_rows, percentile
    from .engine import LANG_CODES
    from .runtime_guard import check_translation
    from .streaming_benchmark import bleu_tokenizer, common_prefix, source_units
    from .prepare_phomt import classify
    from .engine_router import DirectionalNLLBRouter, MODEL_600M, MODEL_1_3B
except ImportError:
    from benchmark import load_rows, percentile
    from engine import LANG_CODES
    from runtime_guard import check_translation
    from streaming_benchmark import bleu_tokenizer, common_prefix, source_units
    from prepare_phomt import classify
    from engine_router import DirectionalNLLBRouter, MODEL_600M, MODEL_1_3B


class CoreTests(unittest.TestCase):
    def test_twelve_directions(self):
        self.assertEqual(len(LANG_CODES) * (len(LANG_CODES) - 1), 12)

    def test_directional_quality_router(self):
        self.assertEqual(DirectionalNLLBRouter.model_for("en", "zh"), MODEL_600M)
        self.assertEqual(DirectionalNLLBRouter.model_for("en", "ko"), MODEL_600M)
        self.assertEqual(DirectionalNLLBRouter.model_for("vi", "en"), MODEL_1_3B)

    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3], .5), 2)

    def test_streaming_helpers(self):
        self.assertEqual(common_prefix(["a", "b"], ["a", "c"]), 1)
        self.assertEqual(source_units("你好", "zh")[0], ["你", "好"])
        self.assertEqual(bleu_tokenizer("zh"), "zh")
        self.assertEqual(bleu_tokenizer("ko"), "13a")

    def test_phomt_file_detection(self):
        self.assertEqual(classify(Path("PhoMT/train.vi")), ("train", "vi"))
        self.assertEqual(classify(Path("PhoMT/dev.en")), ("validation", "en"))

    def test_safety_number_loss(self):
        verdict = check_translation("Pressure is 35 bar", "Áp suất cao", "en", "vi")
        self.assertFalse(verdict["safe"])

    def test_dataset_limit_per_direction(self):
        rows = [{"id": str(i), "src_lang": "vi", "tgt_lang": "en",
                 "source": "x", "reference": "y"} for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            self.assertEqual(len(load_rows(path, 2)), 2)


if __name__ == "__main__":
    unittest.main()
