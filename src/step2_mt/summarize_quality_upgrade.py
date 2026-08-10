"""Turn quality-upgrade artifacts into a conservative deployment decision."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upgrade-dir", type=Path, required=True)
    p.add_argument("--baseline-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    candidate = read_csv(args.upgrade_dir / "flores_nllb_1_3b_summary.csv")
    baseline = read_csv(args.baseline_summary)
    by_direction = {r["direction"]: r for r in baseline}
    comparison = []
    for row in candidate:
        old = by_direction[row["direction"]]
        comparison.append({
            "direction": row["direction"],
            "bleu_600m": float(old["bleu"]), "bleu_1_3b": float(row["bleu"]),
            "bleu_delta": round(float(row["bleu"]) - float(old["bleu"]), 2),
            "chrf_600m": float(old["chrf_pp"]), "chrf_1_3b": float(row["chrf_pp"]),
            "chrf_delta": round(float(row["chrf_pp"]) - float(old["chrf_pp"]), 2),
        })
    profiles = {
        "600m": read_csv(args.upgrade_dir / "gpu_profile_600m.csv"),
        "1_3b": read_csv(args.upgrade_dir / "gpu_profile_1_3b.csv"),
    }
    mean_bleu_delta = sum(r["bleu_delta"] for r in comparison) / len(comparison)
    improved = sum(r["bleu_delta"] > 0 for r in comparison)
    # Require broad, meaningful improvement; otherwise retain the cheaper baseline.
    accept_1_3b = mean_bleu_delta >= 1.0 and improved >= 9
    regressions = [r["direction"] for r in comparison if r["bleu_delta"] <= 0]
    hybrid_bleu_delta = sum(max(0.0, r["bleu_delta"]) for r in comparison) / len(comparison)
    decision = {
        "directions": len(comparison), "mean_bleu_delta": round(mean_bleu_delta, 3),
        "mean_chrf_delta": round(sum(r["chrf_delta"] for r in comparison) / len(comparison), 3),
        "directions_bleu_improved": improved,
        "streaming_files": len(list((args.upgrade_dir / "streaming").glob("*.csv"))),
        "alignatt_files": len(list((args.upgrade_dir / "alignatt").glob("*.csv"))),
        "selected_model": "facebook/nllb-200-1.3B" if accept_1_3b else "facebook/nllb-200-distilled-600M",
        "selection_rule": "1.3B requires mean BLEU delta >= 1.0 and improvement in >= 9/12 directions",
        "recommended_profile": "directional_hybrid",
        "hybrid_1_3b_directions": [r["direction"] for r in comparison if r["bleu_delta"] > 0],
        "hybrid_600m_directions": regressions,
        "hybrid_estimated_mean_bleu_delta": round(hybrid_bleu_delta, 3),
        "hybrid_note": "Use the 600M fallback for measured regressions; no score is extrapolated to unseen data.",
        "comparison": comparison, "gpu_profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
