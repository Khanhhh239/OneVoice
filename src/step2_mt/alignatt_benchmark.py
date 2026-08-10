"""Text-NLLB adaptation of AlignAtt with an auditable quality/latency curve.

The original AlignAtt operates on speech frames. Here, target subwords aligned
to the last ``f`` source tokens are held until more source arrives. Results are
therefore reported in source-token AL/LAAL and must not be presented as seconds.
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


def common_ids(a: list[int], b: list[int]) -> int:
    index = 0
    while index < min(len(a), len(b)) and a[index] == b[index]:
        index += 1
    return index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--src", required=True); p.add_argument("--tgt", required=True)
    p.add_argument("--f", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--read-chunk", type=int, default=4)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [r for r in load_rows(args.dataset, None)
            if r["src_lang"] == args.src and r["tgt_lang"] == args.tgt][:args.limit]
    if not rows:
        raise ValueError("No matching direction")
    engine = NLLBEngine(device=args.device)
    summaries = []
    for hold in args.f:
        hypotheses, references, als, laals, walls, flickers = [], [], [], [], [], []
        for row in rows:
            source_ids = engine.tokenizer(row["source"], add_special_tokens=False).input_ids
            emitted: list[int] = []; write_at: list[int] = []; revisions = 0
            started = time.perf_counter()
            for read in range(args.read_chunk, len(source_ids) + args.read_chunk, args.read_chunk):
                consumed = min(read, len(source_ids))
                prefix = engine.tokenizer.decode(source_ids[:consumed], skip_special_tokens=True)
                tokens, aligned = engine.translate_with_alignment(prefix, args.src, args.tgt)
                shared = common_ids(emitted, tokens)
                if shared < len(emitted):
                    revisions += len(emitted) - shared
                    # Already emitted tokens cannot be retracted in simultaneous MT.
                    if consumed < len(source_ids):
                        continue
                allowed = len(tokens) if consumed == len(source_ids) else 0
                if consumed < len(source_ids):
                    boundary = max(0, consumed - hold)
                    for index, position in enumerate(aligned):
                        if position < boundary:
                            allowed = index + 1
                        else:
                            break
                if tokens[:len(emitted)] == emitted:
                    for token in tokens[len(emitted):allowed]:
                        emitted.append(token); write_at.append(consumed)
                if consumed == len(source_ids):
                    break
            engine.synchronize(); walls.append(time.perf_counter() - started); flickers.append(revisions)
            hypothesis = engine.tokenizer.decode(emitted, skip_special_tokens=True)
            hypotheses.append(hypothesis); references.append(row["reference"])
            target_len, source_len = max(1, len(emitted)), max(1, len(source_ids))
            gamma = target_len / source_len
            lag = [g - i / gamma for i, g in enumerate(write_at)] or [source_len]
            als.append(sum(lag) / len(lag))
            ref_len = max(1, len(engine.tokenizer(row["reference"], add_special_tokens=False).input_ids))
            oracle_gamma = ref_len / source_len
            laal = [g - i / oracle_gamma for i, g in enumerate(write_at)] or [source_len]
            laals.append(sum(laal) / len(laal))
        tokenizer = "zh" if args.tgt == "zh" else "13a"
        summaries.append({"direction": f"{args.src}->{args.tgt}", "policy": "alignatt-text",
            "f_source_tokens": hold, "read_chunk_tokens": args.read_chunk, "n": len(rows),
            "bleu": round(sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenizer).score, 2),
            "chrf_pp": round(sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score, 2),
            "al_source_tokens": round(statistics.mean(als), 3),
            "laal_source_tokens": round(statistics.mean(laals), 3),
            "flicker_tokens_mean": round(statistics.mean(flickers), 3),
            "p50_wall_sec": round(percentile(walls, .5), 4),
            "p95_wall_sec": round(percentile(walls, .95), 4)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys()); writer.writeheader(); writer.writerows(summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
