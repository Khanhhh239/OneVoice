"""Step 1 test data -- fetch labeled (audio, transcript) pairs for ASR
WER/CER testing (test_asr_vi.py / test_asr_multi.py).

Uses Google FLEURS (google/fleurs) for ALL 4 languages -- it's the one
dataset with an identical schema (audio + transcription) across
Vietnamese/English/Chinese/Korean, so there's no guessing 4 different
dataset column names. Streams (no full-archive download) and takes only
N_PER_LANG examples per language.

Separate from data/clean/ (Step 0's VAD/denoise/beamform test audio, which
never needed transcripts) -- writes to data/asr/<lang>/*.wav + manifest.json.
"""
import os
import json

import librosa
import numpy as np
from datasets import load_dataset

from common import SR, save_wav

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "asr")
N_PER_LANG = 5

FLEURS_CONFIG = {
    "vi": "vi_vn",
    "en": "en_us",
    "zh": "cmn_hans_cn",
    "ko": "ko_kr",
}


def fetch_lang(lang, config, n):
    print(f"[fetch_asr_data] {lang}: streaming google/fleurs ({config}) test split...")
    ds = load_dataset("google/fleurs", config, split="test",
                       streaming=True, trust_remote_code=True)
    items = []
    for i, ex in enumerate(ds.take(n)):
        audio = ex["audio"]
        wav = np.asarray(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        text = (ex.get("transcription") or ex.get("raw_transcription") or "").strip()
        if not text:
            continue
        out_path = os.path.join(OUT_DIR, lang, f"{lang}_{i}.wav")
        save_wav(out_path, wav, SR)
        items.append({
            "lang": lang,
            "path": os.path.relpath(out_path, ROOT).replace("\\", "/"),
            "transcript": text,
            "duration_s": round(len(wav) / SR, 3),
        })
    return items


def main():
    manifest = []
    for lang, config in FLEURS_CONFIG.items():
        try:
            items = fetch_lang(lang, config, N_PER_LANG)
            manifest.extend(items)
            print(f"[fetch_asr_data] {lang}: got {len(items)} labeled samples")
        except Exception as e:
            print(f"[fetch_asr_data] {lang}: FAILED ({e!r}) -- skipping")

    if not manifest:
        print("[fetch_asr_data] Nothing fetched -- check internet access / "
              "HF `datasets` install.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[fetch_asr_data] wrote {len(manifest)} labeled samples -> {manifest_path}")


if __name__ == "__main__":
    main()
