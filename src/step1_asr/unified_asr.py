"""Unified ASR Pipeline for OneVoice.
Automatically routes audio to Zipformer-30M (for Vietnamese) 
or SenseVoice-Small (for English/Chinese/Korean).
"""
import os
import sys

# Ensure src/ is in the python path to import from other modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import SR, get_device, load_wav

# Import model loading functions from existing scripts
from step1_asr.test_asr_zipformer import find_model_files
from step1_asr.test_asr_multi import MODEL_ID_HF, MODEL_ID_MODELSCOPE, run_asr as run_sensevoice

class UnifiedASRPipeline:
    def __init__(self):
        self.device = get_device()
        self.device_str = "cuda:0" if self.device.type == "cuda" else "cpu"
        
        print(f"[UnifiedASRPipeline] Initializing on {self.device_str}...")
        
        # Load Zipformer (Vietnamese)
        import sherpa_onnx
        encoder, decoder, joiner, tokens = find_model_files()
        self.zipformer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=2,
            sample_rate=SR,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cuda" if self.device.type == "cuda" else "cpu",
        )
        print("[UnifiedASRPipeline] Zipformer-30M (Vi) loaded.")
        
        # Load SenseVoice (En/Zh/Ko)
        from funasr import AutoModel
        try:
            self.sensevoice = AutoModel(model=MODEL_ID_HF, hub="hf", device=self.device_str)
            print("[UnifiedASRPipeline] SenseVoice-Small (En/Zh/Ko) loaded via HuggingFace.")
        except TypeError:
            self.sensevoice = AutoModel(model=MODEL_ID_MODELSCOPE, device=self.device_str)
            print("[UnifiedASRPipeline] SenseVoice-Small (En/Zh/Ko) loaded via ModelScope.")

    def transcribe(self, path: str, lang: str) -> str:
        """Transcribes the audio file located at `path` using the model appropriate for `lang`."""
        if lang == "vi":
            wav = load_wav(path)
            stream = self.zipformer.create_stream()
            stream.accept_waveform(SR, wav)
            self.zipformer.decode_stream(stream)
            return stream.result.text.strip()
        elif lang in ["en", "zh", "ko"]:
            return run_sensevoice(self.sensevoice, path, lang)
        else:
            raise ValueError(f"Unsupported language requested: {lang}")
