"""Step 1 test -- Moonshine (Vi/En/Zh/Ko ASR, alt candidate to PhoWhisper+
SenseVoice split). Surfaced after a deeper 2026 landscape check (see
step1.md SS12): tiny (~27M param) MONOLINGUAL models per language -- the
"Flavors of Moonshine" paper found monolingual beats multilingual at this
size. Natively supported in `transformers`, so this uses the same pipeline
pattern as test_asr_vi.py. Tests all 4 languages so it can be compared
directly against the current PhoWhisper (Vi) + SenseVoice (En/Zh/Ko) split.
"""
import os
import csv
import json
import time

import torch
import jiwer

from common import SR, get_device, load_wav, rtf, normalize_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASR_DIR = os.path.join(ROOT, "data", "asr")
ASR_MIXED_DIR = os.path.join(ROOT, "data", "asr_mixed")
RESULTS_CSV = os.path.join(ROOT, "outputs", "asr_moonshine_results.csv")

MODEL_IDS = {
    "vi": "UsefulSensors/moonshine-tiny-vi",
    "en": "UsefulSensors/moonshine-tiny",
    "zh": "UsefulSensors/moonshine-tiny-zh",
    "ko": "UsefulSensors/moonshine-tiny-ko",
}
CER_LANGS = {"zh", "ko"}  # word boundaries ambiguous/absent, see step1.md SS2.4


def load_items():
    items = []
    for manifest_path, path_key, default_snr in [
        (os.path.join(ASR_DIR, "manifest.json"), "path", "clean"),
        (os.path.join(ASR_MIXED_DIR, "manifest.json"), "mixed_path", None),
    ]:
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path, "r", encoding="utf-8") as f:
            for it in json.load(f):
                items.append({
                    "lang": it["lang"],
                    "path": it[path_key],
                    "transcript": it["transcript"],
                    "snr_db": it["snr_db"] if default_snr is None else default_snr,
                })
    return items


def main():
    device = get_device()
    print(f"[test_asr_moonshine] device = {device}")

    items = load_items()
    if not items:
        print("[test_asr_moonshine] No ASR data found. Run fetch_asr_data.py "
              "(and mix_asr_noise.py for noisy conditions) first.")
        return

    from transformers import pipeline
    pipes = {}  # lazy-load one pipeline per language, only as needed

    rows = []
    for item in items:
        lang = item["lang"]
        if lang not in MODEL_IDS:
            continue
        if lang not in pipes:
            print(f"[test_asr_moonshine] loading {MODEL_IDS[lang]} for '{lang}'...")
            pipes[lang] = pipeline(
                "automatic-speech-recognition",
                model=MODEL_IDS[lang],
                device=0 if device.type == "cuda" else -1,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )

        path = os.path.join(ROOT, item["path"])
        wav = load_wav(path)
        t0 = time.perf_counter()
        result = pipes[lang]({"raw": wav, "sampling_rate": SR})
        elapsed = time.perf_counter() - t0
        hyp = result["text"].strip()
        ref = item["transcript"]

        metric = "cer" if lang in CER_LANGS else "wer"
        ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
        score = jiwer.cer(ref_n, hyp_n) if metric == "cer" else jiwer.wer(ref_n, hyp_n)
        r = rtf(elapsed, len(wav) / SR)

        row = {
            "lang": lang,
            "file": os.path.basename(item["path"]),
            "snr_db": item["snr_db"],
            "metric": metric,
            "score": round(score, 4),
            "rtf": round(r, 5),
            "device": str(device),
            "reference": ref,
            "hypothesis": hyp,
        }
        rows.append(row)
        print(f"[test_asr_moonshine] {lang} SNR={str(row['snr_db']):>6}  "
              f"{metric.upper()}={score:.3f}  RTF={r:.4f}  hyp='{hyp[:50]}'")

    if not rows:
        print("[test_asr_moonshine] No matching-language items found.")
        return

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[test_asr_moonshine] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
