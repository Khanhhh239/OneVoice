# -*- coding: utf-8 -*-
"""
Piper Vietnamese NPU / CPU TTS Engine
Supports:
1. Singleton model caching (loads weights ONCE into memory, inferences multiple times).
2. Smart Vietnamese sentence segmentation and syllable/byte chunk limiter (max 40 syllables / 512 bytes).
3. Text normalization (Vietnamese number to words, punctuation handling, NFC normalization).
4. Batch / multi-sentence inference with per-sentence metrics (Latency, Audio Duration, RTF, OOV, Waveform).
5. Full combined WAV generation.
6. Side-by-side comparison with Reference Piper ONNX.
"""

import os
import sys
import re
import time
import io
import wave
import unicodedata
from typing import List, Dict, Any, Tuple, Optional

# Ensure OpenMP runtime conflict is suppressed on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

HERE = os.path.dirname(os.path.abspath(__file__))
PIPER_NPU_DIR = os.path.join(HERE, "piper_npu")
G2P_DIR = os.path.join(PIPER_NPU_DIR, "g2p")
STEP4_DIR = os.path.join(PIPER_NPU_DIR, "src", "step4_npu")

for p in [HERE, PIPER_NPU_DIR, G2P_DIR, STEP4_DIR]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

import numpy as np
import torch
import full_graph as F
import g2p_graph as G
from piper import PiperVoice

SAMPLE_RATE = 22050
DEFAULT_MAX_SYLLABLES = 38
MAX_BYTES = G.LB  # 512


# -------------------------------------------------------------
# Vietnamese Number to Words Converter
# -------------------------------------------------------------
DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
UNITS = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ"]

def _read_3digits(num: int, has_high: bool = False) -> str:
    h = num // 100
    t = (num % 100) // 10
    u = num % 10
    res = []
    if h > 0 or has_high:
        res.append(DIGITS[h] + " trăm")
    if t == 0:
        if (h > 0 or has_high) and u > 0:
            res.append("lẻ")
    elif t == 1:
        res.append("mười")
    else:
        res.append(DIGITS[t] + " mươi")
    if u > 0:
        if t > 0 and u == 1:
            res.append("mốt")
        elif t > 0 and u == 5:
            res.append("lăm")
        elif t == 0 and u == 5 and (h > 0 or has_high):
            res.append("năm")
        else:
            res.append(DIGITS[u])
    return " ".join(res)

def num_to_vietnamese_words(num_str: str) -> str:
    """Convert number string to Vietnamese spoken words."""
    try:
        n = int(num_str.replace(".", "").replace(",", ""))
    except ValueError:
        return num_str
    if n == 0:
        return "không"
    groups = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000
    res = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g > 0:
            has_high = (i < len(groups) - 1)
            words = _read_3digits(g, has_high=has_high)
            unit = UNITS[i % len(UNITS)]
            if unit:
                words += " " + unit
            res.append(words)
    return " ".join(res)

def normalize_vietnamese_text(text: str) -> str:
    """Normalize text: NFC, convert digits, clean non-standard spaces/punctuations."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text.strip())
    # Replace numbers with Vietnamese words
    text = re.sub(r'\b\d+\b', lambda m: num_to_vietnamese_words(m.group(0)), text)
    # Replace percent symbol
    text = text.replace("%", " phần trăm ")
    # Normalize dashes and quotes
    text = text.replace("–", "-").replace("—", "-").replace('“', '"').replace('”', '"')
    # Clean whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# -------------------------------------------------------------
# Smart Sentence Splitter & Chunk Limiter
# -------------------------------------------------------------
def split_into_sentences(text: str, max_syllables: int = DEFAULT_MAX_SYLLABLES) -> List[str]:
    """
    Split text into individual sentences and ensure each sentence stays within max_syllables.
    Avoids overflowing the NPU static input buffer (512 bytes / ~40 syllables).
    """
    text = normalize_vietnamese_text(text)
    if not text:
        return []

    # First split by major sentence boundaries (. ! ? \n ;)
    raw_sentences = re.split(r'([.!?\n;]+)', text)
    merged_sents = []
    curr = ""
    for piece in raw_sentences:
        if re.match(r'^[.!?\n;]+$', piece):
            curr += piece
            if curr.strip():
                merged_sents.append(curr.strip())
            curr = ""
        else:
            curr += piece
    if curr.strip():
        merged_sents.append(curr.strip())

    # Now verify syllable and byte length limit per sentence
    final_chunks = []
    for sent in merged_sents:
        words = sent.split()
        if not words:
            continue

        # If sentence is within limits, keep as is
        encoded_len = len(sent.encode("utf-8"))
        if len(words) <= max_syllables and encoded_len <= (MAX_BYTES - 20):
            final_chunks.append(sent)
            continue

        # Otherwise sub-split on commas or word boundaries
        sub_chunks = []
        clause_parts = re.split(r'([,:\(\)])', sent)
        accum = ""
        for part in clause_parts:
            test_accum = (accum + " " + part).strip() if accum else part.strip()
            test_words = test_accum.split()
            test_bytes = len(test_accum.encode("utf-8"))
            if len(test_words) <= max_syllables and test_bytes <= (MAX_BYTES - 20):
                accum = test_accum
            else:
                if accum.strip():
                    sub_chunks.append(accum.strip())
                accum = part.strip()
        if accum.strip():
            sub_chunks.append(accum.strip())

        # If any sub-chunk is still too long, do strict word slicing
        for chunk in sub_chunks:
            chunk_words = chunk.split()
            if len(chunk_words) <= max_syllables:
                final_chunks.append(chunk)
            else:
                for i in range(0, len(chunk_words), max_syllables):
                    slice_words = chunk_words[i : i + max_syllables]
                    final_chunks.append(" ".join(slice_words))

    # Add terminal punctuation if missing to ensure natural intonation
    cleaned_chunks = []
    for s in final_chunks:
        s = s.strip()
        if s and not s[-1] in ".!?,;:":
            s += "."
        if s:
            cleaned_chunks.append(s)

    return cleaned_chunks


# -------------------------------------------------------------
# Singleton TTS Model Manager
# -------------------------------------------------------------
class PiperTTSModelManager:
    _instance = None

    def __init__(self):
        self.onnx_path = F.ONNX
        self.config_path = F.ONNX + ".json"
        self.full_model = None
        self.g2p_table = None
        self.ref_voice = None
        self.is_loaded = False
        self._load_models()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = PiperTTSModelManager()
        return cls._instance

    def _load_models(self):
        """Loads FullTTS PyTorch single graph and PiperVoice ONNX reference."""
        t0 = time.time()
        print(f"[TTS Engine] Loading FullTTS static graph model...")
        self.full_model, self.g2p_table = F.build(group=8)
        self.full_model.eval()

        print(f"[TTS Engine] Loading PiperVoice reference ONNX...")
        self.ref_voice = PiperVoice.load(self.onnx_path, config_path=self.config_path)
        self.is_loaded = True
        t_load = time.time() - t0
        print(f"[TTS Engine] Models loaded successfully in {t_load:.2f}s!")

    def synthesize_sentence_full(
        self,
        text: str,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Synthesize a single sentence via FullTTS (Single Static Graph).
        Returns audio (numpy float32), sample_rate, latency_ms, audio_duration_s, rtf, oov, y_len.
        """
        t0 = time.perf_counter()
        raw_bytes = F.text_to_bytes(text)
        b_tensor = torch.from_numpy(raw_bytes)

        # Update speed / noise buffers
        self.full_model.ls.copy_(torch.tensor([length_scale], dtype=torch.float32))
        self.full_model.nw.copy_(torch.tensor([noise_w], dtype=torch.float32))
        self.full_model.ns.copy_(torch.tensor([noise_scale], dtype=torch.float32))

        with torch.no_grad():
            audio_out, y_len_out, oov_out = self.full_model(b_tensor)

        t_infer = time.perf_counter() - t0
        y_len = int(y_len_out[0].item())
        n_samples = y_len * F.UP  # HOP * UP = 256
        audio_np = audio_out[0, :n_samples].cpu().numpy().astype(np.float32)

        # Normalize audio volume to avoid clipping
        max_val = np.max(np.abs(audio_np)) if len(audio_np) > 0 else 0
        if max_val > 0.99:
            audio_np = audio_np / max_val * 0.95

        dur_s = len(audio_np) / SAMPLE_RATE if SAMPLE_RATE > 0 else 0.0
        rtf = (t_infer / dur_s) if dur_s > 0 else 0.0
        oov_count = int(oov_out[0].item())

        return {
            "text": text,
            "audio": audio_np,
            "sample_rate": SAMPLE_RATE,
            "latency_ms": round(t_infer * 1000, 2),
            "duration_s": round(dur_s, 3),
            "rtf": round(rtf, 4),
            "y_len": y_len,
            "oov_count": oov_count,
            "mode": "FullTTS (Single Graph)",
        }

    def synthesize_sentence_ref(self, text: str) -> Dict[str, Any]:
        """Synthesize using reference PiperVoice (eSpeak G2P + ONNX)."""
        t0 = time.perf_counter()
        raw_chunks = []
        for chunk in self.ref_voice.synthesize(text):
            if hasattr(chunk, "audio_int16_bytes"):
                raw_chunks.append(chunk.audio_int16_bytes)
            elif isinstance(chunk, bytes):
                raw_chunks.append(chunk)
        audio_bytes = b"".join(raw_chunks)
        t_infer = time.perf_counter() - t0

        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_np = audio_int16.astype(np.float32) / 32767.0
        dur_s = len(audio_np) / SAMPLE_RATE if SAMPLE_RATE > 0 else 0.0
        rtf = (t_infer / dur_s) if dur_s > 0 else 0.0

        return {
            "text": text,
            "audio": audio_np,
            "sample_rate": SAMPLE_RATE,
            "latency_ms": round(t_infer * 1000, 2),
            "duration_s": round(dur_s, 3),
            "rtf": round(rtf, 4),
            "mode": "Piper Reference (eSpeak+ONNX)",
        }

    def batch_synthesize(
        self,
        text: str,
        max_syllables: int = DEFAULT_MAX_SYLLABLES,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        compare_with_ref: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a multi-sentence text:
        1. Segments into valid chunks.
        2. Inferences sequentially reusing the loaded model weights.
        3. Computes metrics per sentence + aggregates total audio and statistics.
        """
        sentences = split_into_sentences(text, max_syllables=max_syllables)
        if not sentences:
            return {
                "error": "Không tìm thấy nội dung văn bản hợp lệ sau khi xử lý.",
                "sentences": [],
                "results": [],
            }

        sentence_results = []
        all_audio_chunks = []
        ref_audio_chunks = []
        total_infer_time = 0.0
        total_oov = 0

        # Silence gap between sentences (0.15s)
        silence = np.zeros(int(SAMPLE_RATE * 0.15), dtype=np.float32)

        for i, sent in enumerate(sentences):
            res = self.synthesize_sentence_full(
                sent,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
            )
            total_infer_time += res["latency_ms"] / 1000.0
            total_oov += res["oov_count"]

            if compare_with_ref:
                ref_res = self.synthesize_sentence_ref(sent)
                res["ref_audio"] = ref_res["audio"]
                res["ref_latency_ms"] = ref_res["latency_ms"]
                # Cosine similarity between full and ref (truncated to min len)
                min_len = min(len(res["audio"]), len(ref_res["audio"]))
                if min_len > 0:
                    a1 = res["audio"][:min_len]
                    a2 = ref_res["audio"][:min_len]
                    norm1, norm2 = np.linalg.norm(a1), np.linalg.norm(a2)
                    cos_sim = float(np.dot(a1, a2) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
                else:
                    cos_sim = 0.0
                res["cosine_sim"] = round(cos_sim, 4)
                ref_audio_chunks.append(ref_res["audio"])

            sentence_results.append(res)
            all_audio_chunks.append(res["audio"])

        # Combine all audio chunks with short silence between them
        combined_audio_list = []
        for idx, chunk in enumerate(all_audio_chunks):
            combined_audio_list.append(chunk)
            if idx < len(all_audio_chunks) - 1:
                combined_audio_list.append(silence)
        full_audio = np.concatenate(combined_audio_list) if combined_audio_list else np.array([], dtype=np.float32)

        total_audio_dur = len(full_audio) / SAMPLE_RATE if len(full_audio) > 0 else 0.0
        overall_rtf = (total_infer_time / total_audio_dur) if total_audio_dur > 0 else 0.0

        return {
            "raw_text": text,
            "num_sentences": len(sentences),
            "sentences": sentences,
            "results": sentence_results,
            "full_audio": full_audio,
            "sample_rate": SAMPLE_RATE,
            "total_audio_duration_s": round(total_audio_dur, 3),
            "total_inference_time_s": round(total_infer_time, 3),
            "overall_rtf": round(overall_rtf, 4),
            "total_oov_count": total_oov,
        }


def audio_to_wav_bytes(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert numpy float32 audio to WAV bytes."""
    buf = io.BytesIO()
    int16_audio = np.clip(audio_np * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_audio.tobytes())
    return buf.getvalue()
