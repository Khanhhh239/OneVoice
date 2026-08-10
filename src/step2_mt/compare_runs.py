"""Create an auditable before/after table from two benchmark summary CSVs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {row["direction"]: row for row in csv.DictReader(f)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    before, after = read(args.before), read(args.after)
    directions = sorted(before.keys() & after.keys())
    if not directions:
        raise ValueError("The summaries have no common directions")
    rows = []
    for direction in directions:
        b, a = before[direction], after[direction]
        rows.append({"direction": direction, "n_before": b["n"], "n_after": a["n"],
                     "bleu_before": b["bleu"], "bleu_after": a["bleu"],
                     "bleu_delta": round(float(a["bleu"]) - float(b["bleu"]), 2),
                     "chrf_before": b["chrf_pp"], "chrf_after": a["chrf_pp"],
                     "chrf_delta": round(float(a["chrf_pp"]) - float(b["chrf_pp"]), 2),
                     "p50_before": b["p50_sec_per_sentence"],
                     "p50_after": a["p50_sec_per_sentence"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
