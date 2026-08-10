"""Interactive OneVoice MT demo with a reference-free safety guard."""
import argparse
import json
import time

try:
    from .engine import LANG_CODES, NLLBEngine
    from .runtime_guard import check_translation
except ImportError:
    from engine import LANG_CODES, NLLBEngine
    from runtime_guard import check_translation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mode", choices=["live", "accurate"], default="live")
    args = parser.parse_args()
    beams = 1 if args.mode == "live" else 4
    engine = NLLBEngine(device=args.device)
    print("OneVoice MT ready. Languages: vi, en, zh, ko. Type 'quit' to stop.")
    while True:
        src = input("Source language: ").strip().lower()
        if src == "quit":
            break
        tgt = input("Target language: ").strip().lower()
        text = input("Text: ").strip()
        if src not in LANG_CODES or tgt not in LANG_CODES or src == tgt or not text:
            print("Invalid input. Use distinct vi/en/zh/ko languages and non-empty text.")
            continue
        started = time.perf_counter()
        translation = engine.translate_batch([text], src, tgt, num_beams=beams)[0]
        engine.synchronize()
        elapsed = time.perf_counter() - started
        guard = check_translation(text, translation, src, tgt)
        print(f"Translation: {translation}")
        print(f"Latency: {elapsed:.3f}s | safety_guard={'PASS' if guard['safe'] else 'REVIEW'}")
        if guard["warnings"]:
            print("Warnings:", json.dumps(guard["warnings"], ensure_ascii=False))


if __name__ == "__main__":
    main()
