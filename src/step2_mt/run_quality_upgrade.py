"""Resumable Kaggle validation that closes the remaining Step-2 evidence gaps."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOD = "src.step2_mt"


def run(arguments: list[str], expected: Path | None = None) -> None:
    if expected and expected.exists() and expected.stat().st_size:
        print(f"[skip] {expected}", flush=True)
        return
    print("[run]", " ".join(arguments), flush=True)
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "outputs/mt/quality_upgrade")
    p.add_argument("--stream-limit", type=int, default=20)
    p.add_argument("--align-limit", type=int, default=10)
    p.add_argument("--skip-comet", action="store_true")
    p.add_argument("--skip-alignatt", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    flores = ROOT / "data/mt/flores_all_pairs.jsonl"
    if not flores.exists():
        raise FileNotFoundError(flores)

    # Full quality comparison against the already-tested 600M baseline.
    details = args.output / "flores_nllb_1_3b_details.jsonl"
    run(["-m", f"{MOD}.benchmark", "--dataset", str(flores),
         "--model", "facebook/nllb-200-1.3B", "--device", "cuda",
         "--batch-size", "16", "--output-dir", str(args.output),
         "--run-name", "flores_nllb_1_3b"], details)
    metrics = args.output / "metrics/flores_nllb_1_3b_metrics_details.jsonl"
    run(["-m", f"{MOD}.score_metrics", "--details", str(details),
         "--skip-comet", "--output-dir", str(args.output / "metrics")], metrics)
    if not args.skip_comet:
        comet = args.output / "comet/flores_nllb_1_3b_metrics_details.jsonl"
        run(["-m", f"{MOD}.run_comet_isolated", "--details", str(details),
             "--output-dir", str(args.output / "comet"), "--batch-size", "8"], comet)

    # Comparable GPU operational profile; subprocesses release VRAM between models.
    for label, model in (("600m", "facebook/nllb-200-distilled-600M"),
                         ("1_3b", "facebook/nllb-200-1.3B")):
        output = args.output / f"gpu_profile_{label}.csv"
        run(["-m", f"{MOD}.profile_gpu", "--dataset", str(flores),
             "--model", model, "--output", str(output)], output)

    languages = ("vi", "en", "zh", "ko")
    for src in languages:
        for tgt in languages:
            if src == tgt:
                continue
            output = args.output / "streaming" / f"streaming_{src}_{tgt}.csv"
            run(["-m", f"{MOD}.streaming_benchmark", "--dataset", str(flores),
                 "--src", src, "--tgt", tgt, "--chunk-tokens", "2", "4", "8",
                 "--limit", str(args.stream_limit), "--device", "cuda",
                 "--output", str(output)], output)
            if not args.skip_alignatt:
                align = args.output / "alignatt" / f"alignatt_{src}_{tgt}.csv"
                run(["-m", f"{MOD}.alignatt_benchmark", "--dataset", str(flores),
                     "--src", src, "--tgt", tgt, "--f", "1", "2", "4",
                     "--read-chunk", "4", "--limit", str(args.align_limit),
                     "--device", "cuda", "--output", str(align)], align)

    baseline_comet = ROOT / "outputs/mt/comet/flores_metrics_details.jsonl"
    if baseline_comet.exists():
        run(["-m", f"{MOD}.extract_worst_cases", "--details", str(baseline_comet),
             "--per-direction", "30", "--output", str(args.output / "human_review_360.csv")],
            args.output / "human_review_360.csv")
    print(f"QUALITY UPGRADE COMPLETE: {args.output}")


if __name__ == "__main__":
    main()
