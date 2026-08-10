"""Reproducible corpus-level MT benchmark for JSONL reference datasets."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import sacrebleu

try:
    from .engine import NLLBEngine
except ImportError:  # direct script execution
    from engine import NLLBEngine

ROOT = Path(__file__).resolve().parents[2]


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def load_rows(path: Path, limit_per_direction: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    required = {"id", "src_lang", "tgt_lang", "source", "reference"}
    selected, counts = [], defaultdict(int)
    for index, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {index} missing fields: {sorted(missing)}")
        direction = (row["src_lang"], row["tgt_lang"])
        if limit_per_direction is None or counts[direction] < limit_per_direction:
            selected.append(row)
            counts[direction] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--backend", choices=["pytorch", "ct2"], default="pytorch")
    parser.add_argument("--tokenizer", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--limit-per-direction", type=int)
    parser.add_argument("--src", choices=["vi", "en", "zh", "ko"])
    parser.add_argument("--tgt", choices=["vi", "en", "zh", "ko"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mt")
    parser.add_argument("--run-name", help="Unique output stem, e.g. flores_base or flores_lora")
    args = parser.parse_args()

    rows = load_rows(args.dataset, args.limit_per_direction)
    if args.src:
        rows = [row for row in rows if row["src_lang"] == args.src]
    if args.tgt:
        rows = [row for row in rows if row["tgt_lang"] == args.tgt]
    if not rows:
        raise SystemExit("Dataset is empty")
    if args.backend == "ct2":
        try:
            from .engine_ct2 import CT2NLLBEngine
        except ImportError:
            from engine_ct2 import CT2NLLBEngine
        ct2_device = "cuda" if args.device == "cuda" else "cpu"
        engine = CT2NLLBEngine(args.model, args.tokenizer, ct2_device, args.compute_type)
    else:
        engine = NLLBEngine(args.model, args.device)
    for _ in range(args.warmup):
        engine.translate_batch(["Xin chào."], "vi", "en", max_new_tokens=16)
    engine.synchronize()

    groups = defaultdict(list)
    for row in rows:
        groups[(row["src_lang"], row["tgt_lang"])].append(row)
    details = []
    for (src, tgt), group in sorted(groups.items()):
        print(f"[{src}->{tgt}] {len(group)} sentences", flush=True)
        for start in range(0, len(group), args.batch_size):
            chunk = group[start:start + args.batch_size]
            engine.synchronize()
            started = time.perf_counter()
            hypotheses = engine.translate_batch(
                [item["source"] for item in chunk], src, tgt, num_beams=args.num_beams)
            engine.synchronize()
            batch_seconds = time.perf_counter() - started
            for row, hypothesis in zip(chunk, hypotheses):
                details.append({**row, "hypothesis": hypothesis,
                                "amortized_latency_sec": batch_seconds / len(chunk)})

    summaries = []
    for (src, tgt), group in sorted(groups.items()):
        part = [row for row in details if row["src_lang"] == src and row["tgt_lang"] == tgt]
        hypotheses = [row["hypothesis"] for row in part]
        references = [row["reference"] for row in part]
        latencies = [row["amortized_latency_sec"] for row in part]
        tokenizer = "zh" if tgt == "zh" else "13a"
        summaries.append({
            "direction": f"{src}->{tgt}", "n": len(part),
            "bleu": round(sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenizer).score, 2),
            "chrf_pp": round(sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score, 2),
            "mean_sec_per_sentence": round(statistics.mean(latencies), 4),
            "p50_sec_per_sentence": round(percentile(latencies, .50), 4),
            "p95_sec_per_sentence": round(percentile(latencies, .95), 4),
            "throughput_sentences_sec": round(len(part) / sum(latencies), 3),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.run_name or args.dataset.stem
    detail_path = args.output_dir / f"{stem}_details.jsonl"
    detail_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details), encoding="utf-8")
    summary_path = args.output_dir / f"{stem}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"details={detail_path}\nsummary={summary_path}")


if __name__ == "__main__":
    main()
