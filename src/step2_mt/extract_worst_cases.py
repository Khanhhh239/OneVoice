"""Create an auditable per-direction review set from sentence-level COMET."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--details", type=Path, required=True)
    p.add_argument("--per-direction", type=int, default=30)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    groups = defaultdict(list)
    with args.details.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "comet_score" not in row:
                raise SystemExit("Input must contain sentence-level comet_score")
            groups[f'{row["src_lang"]}->{row["tgt_lang"]}'].append(row)
    selected = []
    for direction, rows in sorted(groups.items()):
        for rank, row in enumerate(sorted(rows, key=lambda x: x["comet_score"])[:args.per_direction], 1):
            selected.append({
                "direction": direction, "rank": rank, "id": row["id"],
                "comet_score": round(float(row["comet_score"]), 6),
                "safety_pass": row.get("safety_pass"), "source": row["source"],
                "reference": row["reference"], "hypothesis": row["hypothesis"],
                "human_error_category": "", "human_notes": "", "human_accept": "",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected[0].keys())
        writer.writeheader(); writer.writerows(selected)
    print(f"{args.output}: {len(selected)} rows, {len(groups)} directions")


if __name__ == "__main__":
    main()
