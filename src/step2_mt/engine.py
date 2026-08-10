"""Reusable NLLB engine for all 12 directions among vi/en/zh/ko."""
from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LANG_CODES = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}


class NLLBEngine:
    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M", device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()

    @torch.inference_mode()
    def translate_batch(self, texts: list[str], src: str, tgt: str, *, num_beams: int = 1,
                        max_new_tokens: int = 256) -> list[str]:
        if src not in LANG_CODES or tgt not in LANG_CODES or src == tgt:
            raise ValueError(f"Unsupported direction: {src}->{tgt}")
        self.tokenizer.src_lang = LANG_CODES[src]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(LANG_CODES[tgt]),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def translate_with_alignment(self, text: str, src: str, tgt: str,
                                 max_new_tokens: int = 256) -> tuple[list[int], list[int]]:
        """Return generated token IDs and encoder positions from cross-attention maxima.

        A second teacher-forced forward pass gives a dense [target, source]
        attention matrix. The final four decoder layers and all heads are
        averaged to reduce single-head noise.
        """
        self.tokenizer.src_lang = LANG_CODES[src]
        # PyTorch SDPA deliberately does not return attention tensors.
        # Switch only alignment runs to the eager implementation.
        if hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")
        else:
            self.model.config._attn_implementation = "eager"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        sequence = self.model.generate(
            **inputs, forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(LANG_CODES[tgt]),
            max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
        forward = self.model(
            **inputs, decoder_input_ids=sequence[:, :-1], output_attentions=True,
            return_dict=True, use_cache=False)
        layers = forward.cross_attentions[-4:]
        # Each layer is [batch, heads, target_steps, source_steps].
        attention = torch.stack(layers).mean(dim=(0, 2))[0]
        alignments = attention.argmax(dim=-1).tolist()
        generated = sequence[0, 1:].tolist()
        special = {self.tokenizer.pad_token_id, self.tokenizer.eos_token_id}
        pairs = [(token, pos) for token, pos in zip(generated, alignments) if token not in special]
        return [p[0] for p in pairs], [p[1] for p in pairs]
