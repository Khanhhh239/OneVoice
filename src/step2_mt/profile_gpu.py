"""Measure model load, warm latency, throughput, and peak CUDA memory."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from .benchmark import load_rows, percentile
from .engine import NLLBEngine


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for this profile")

    rows = load_rows(args.dataset, None)
    samples = {}
    for row in rows:
        samples.setdefault((row["src_lang"], row["tgt_lang"]), row["source"])

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    engine = NLLBEngine(args.model, "cuda")
    engine.synchronize()
    load_seconds = time.perf_counter() - started
    model_memory_mib = torch.cuda.memory_allocated() / 1024**2

    results = []
    for batch_size in args.batch_sizes:
        timings = []
        torch.cuda.reset_peak_memory_stats()
        for (src, tgt), text in sorted(samples.items()):
            batch = [text] * batch_size
            engine.translate_batch(batch, src, tgt, max_new_tokens=128)
            engine.synchronize()
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                engine.translate_batch(batch, src, tgt, max_new_tokens=128)
                engine.synchronize()
                timings.append((time.perf_counter() - t0) / batch_size)
        results.append({
            "model": args.model, "batch_size": batch_size,
            "directions": len(samples), "repeats": args.repeats,
            "load_seconds": round(load_seconds, 3),
            "model_memory_mib": round(model_memory_mib, 1),
            "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "mean_sec_sentence": round(statistics.mean(timings), 4),
            "p50_sec_sentence": round(percentile(timings, .50), 4),
            "p95_sec_sentence": round(percentile(timings, .95), 4),
            "p99_sec_sentence": round(percentile(timings, .99), 4),
            "throughput_sentences_sec": round(1 / statistics.mean(timings), 2),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
