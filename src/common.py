"""Shared helpers for Step-0 (Audio Front-end) tests: VAD, denoise, beamform."""
import os
import glob
import time

import numpy as np
import soundfile as sf
import librosa
import torch

SR = 16000
AUDIO_EXT = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_audio_files(folder):
    files = []
    if not os.path.isdir(folder):
        return files
    for ext in AUDIO_EXT:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(files)


def load_wav(path, sr=SR):
    """Load audio as mono float32 at target sample rate."""
    wav, orig_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


def save_wav(path, wav, sr=SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    peak = np.max(np.abs(wav)) if len(wav) else 0.0
    if peak > 0.99:
        wav = wav / peak * 0.99
    sf.write(path, wav.astype(np.float32), sr)


class Timer:
    """`with Timer() as t: ...` then read t.elapsed (seconds)."""
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0


def rtf(process_seconds, audio_seconds):
    """Real-Time Factor = processing time / audio duration. <1 = faster than real-time."""
    return process_seconds / max(audio_seconds, 1e-9)
