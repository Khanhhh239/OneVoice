"""Re-decode streaming baseline with measurable quality/latency trade-off.

This is an honest wait-k/chunk baseline, not an AlignAtt implementation.  It emits
only the stable prefix shared by two consecutive hypotheses and reports token AL
and LAAL-style lag proxies for reproducible comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import sacrebleu

try:
    from .benchmark import load_rows, percentile
    from .engine import NLLBEngine
except ImportError:
    from benchmark import load_rows, percentile
    from engine import NLLBEngine


def common_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def source_units(text: str, lang: str) -> tuple[list[str], str]:
    """Use characters for unsegmented Chinese and words elsewhere."""
    if lang == "zh":
        return list(text), ""
    return text.split(), " "


def bleu_tokenizer(target_language: str) -> str:
    return "zh" if target_language == "zh" else "13a"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--chunk-tokens", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [r for r in load_rows(args.dataset, None)
            if r["src_lang"] == args.src and r["tgt_lang"] == args.tgt][:args.limit]
    if not rows:
        raise ValueError("No matching direction")
    engine = NLLBEngine(device=args.device)
    results = []
    for chunk_size in args.chunk_tokens:
        hyps, refs, latencies, als, laals = [], [], [], [], []
        for row in rows:
            src_tokens, separator = source_units(row["source"], args.src)
            previous, emitted, write_positions = [], [], []
            started = time.perf_counter()
            for read in range(chunk_size, len(src_tokens) + chunk_size, chunk_size):
                consumed = min(read, len(src_tokens))
                hypothesis = engine.translate_batch(
                    [separator.join(src_tokens[:consumed])], args.src, args.tgt)[0].split()
                stable = common_prefix(previous, hypothesis) if previous else 0
                if consumed == len(src_tokens):
                    stable = len(hypothesis)
                for token in hypothesis[len(emitted):stable]:
                    emitted.append(token)
                    write_positions.append(consumed)
                previous = hypothesis
                if consumed == len(src_tokens):
                    break
            engine.synchronize()
            latencies.append(time.perf_counter() - started)
            text = " ".join(emitted)
            hyps.append(text); refs.append(row["reference"])
            target_len = max(1, len(emitted))
            source_len = max(1, len(src_tokens))
            gamma = target_len / source_len
            lag = [g - (i / gamma) for i, g in enumerate(write_positions)] or [source_len]
            als.append(sum(lag) / len(lag))
            oracle_gamma = max(1, len(row["reference"].split())) / source_len
            laal_lag = [g - (i / oracle_gamma) for i, g in enumerate(write_positions)] or [source_len]
            laals.append(sum(laal_lag) / len(laal_lag))
        tokenizer = bleu_tokenizer(args.tgt)
        results.append({
            "direction": f"{args.src}->{args.tgt}", "chunk_tokens": chunk_size,
            "n": len(rows),
            "bleu": round(sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenizer).score, 2),
            "chrf_pp": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
            "al_source_tokens": round(statistics.mean(als), 3),
            "laal_source_tokens": round(statistics.mean(laals), 3),
            "p50_wall_sec": round(percentile(latencies, .5), 4),
            "p95_wall_sec": round(percentile(latencies, .95), 4),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys()); writer.writeheader(); writer.writerows(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
