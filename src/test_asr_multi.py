"""Step 1 test -- SenseVoice-Small (English / Chinese / Korean ASR).
Loads the OFFICIAL pretrained checkpoint from FunAudioLLM via the `funasr`
library -- zero fine-tuning, as decided in step1.md. Measures WER (English)
/ CER (Chinese, Korean -- word boundaries are ambiguous or absent, see
step1.md SS2.4) and RTF on clean + noisy (SNR-mixed) speech.
Maps to Technical Proposal SS4.2/SS4.3 (ASR row) for En/Zh/Ko.

funasr's API has shifted across versions (hub= kwarg, use_itn= kwarg) --
this script tries the documented modern call and falls back to an older
form on TypeError instead of hard-failing.
"""
import os
import csv
import json
import time

import jiwer

from common import SR, get_device, load_wav, rtf, normalize_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASR_DIR = os.path.join(ROOT, "data", "asr")
ASR_MIXED_DIR = os.path.join(ROOT, "data", "asr_mixed")
RESULTS_CSV = os.path.join(ROOT, "outputs", "asr_multi_results.csv")

MODEL_ID_HF = "FunAudioLLM/SenseVoiceSmall"
MODEL_ID_MODELSCOPE = "iic/SenseVoiceSmall"
LANGS = ["en", "zh", "ko"]
CER_LANGS = {"zh", "ko"}  # character-based scoring; "en" uses WER


def load_items():
    items = []
    clean_manifest = os.path.join(ASR_DIR, "manifest.json")
    if os.path.exists(clean_manifest):
        with open(clean_manifest, "r", encoding="utf-8") as f:
            for it in json.load(f):
                if it["lang"] in LANGS:
                    items.append({"lang": it["lang"], "path": it["path"],
                                  "transcript": it["transcript"], "snr_db": "clean"})
    mixed_manifest = os.path.join(ASR_MIXED_DIR, "manifest.json")
    if os.path.exists(mixed_manifest):
        with open(mixed_manifest, "r", encoding="utf-8") as f:
            for it in json.load(f):
                if it["lang"] in LANGS:
                    items.append({"lang": it["lang"], "path": it["mixed_path"],
                                  "transcript": it["transcript"], "snr_db": it["snr_db"]})
    return items


def load_sensevoice(device_str):
    from funasr import AutoModel
    try:
        return AutoModel(model=MODEL_ID_HF, trust_remote_code=True,
                          hub="hf", device=device_str)
    except TypeError:
        print("[test_asr_multi] funasr version doesn't accept hub='hf' -- "
              f"falling back to ModelScope id '{MODEL_ID_MODELSCOPE}'")
        return AutoModel(model=MODEL_ID_MODELSCOPE, trust_remote_code=True,
                          device=device_str)


def run_asr(model, path, lang):
    try:
        result = model.generate(input=path, language=lang, use_itn=True)
    except TypeError:
        result = model.generate(input=path, language=lang)
    if isinstance(result, list) and result and "text" in result[0]:
        return result[0]["text"].strip()
    return str(result).strip()


def main():
    device = get_device()
    device_str = "cuda:0" if device.type == "cuda" else "cpu"
    print(f"[test_asr_multi] device = {device_str}, model = {MODEL_ID_HF}")

    items = load_items()
    if not items:
        print("[test_asr_multi] No En/Zh/Ko ASR data found. Run fetch_asr_data.py "
              "(and mix_asr_noise.py for noisy conditions) first.")
        return

    model = load_sensevoice(device_str)

    rows = []
    for item in items:
        path = os.path.join(ROOT, item["path"])
        wav = load_wav(path)
        t0 = time.perf_counter()
        hyp = run_asr(model, path, item["lang"])
        elapsed = time.perf_counter() - t0
        ref = item["transcript"]

        metric = "cer" if item["lang"] in CER_LANGS else "wer"
        ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
        score = jiwer.cer(ref_n, hyp_n) if metric == "cer" else jiwer.wer(ref_n, hyp_n)
        r = rtf(elapsed, len(wav) / SR)

        row = {
            "lang": item["lang"],
            "file": os.path.basename(item["path"]),
            "snr_db": item["snr_db"],
            "metric": metric,
            "score": round(score, 4),
            "rtf": round(r, 5),
            "device": device_str,
            "reference": ref,
            "hypothesis": hyp,
        }
        rows.append(row)
        print(f"[test_asr_multi] {row['lang']} SNR={str(row['snr_db']):>6}  "
              f"{metric.upper()}={score:.3f}  RTF={r:.4f}  hyp='{hyp[:50]}'")

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[test_asr_multi] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
