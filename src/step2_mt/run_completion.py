"""Resumable Step-2 experiment runner for Kaggle.

Run from the OneVoice root. Completed output files are skipped, so a Kaggle
restart does not require repeating successful stages.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOD = "src.step2_mt"


def run(args: list[str], expected: Path | None = None) -> None:
    if expected and expected.exists() and expected.stat().st_size:
        print(f"[skip] {expected}", flush=True)
        return
    print("[run]", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-streaming", action="store_true")
    p.add_argument("--skip-alignatt", action="store_true")
    p.add_argument("--stream-limit", type=int, default=100)
    p.add_argument("--alignatt-limit", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.output is None:
        args.output = ROOT / "outputs" / "mt" / ("smoke_kaggle_v6" if args.smoke else "full_kaggle")
    args.output.mkdir(parents=True, exist_ok=True)

    run(["-m", f"{MOD}.validate_mvp"])
    datasets = {
        "flores": ROOT / "data" / "mt" / "flores_all_pairs.jsonl",
        "ntrex": ROOT / "data" / "mt" / "ntrex_all_pairs.jsonl",
        "massive": ROOT / "data" / "mt" / "massive_test_all_pairs.jsonl",
    }
    for name, dataset in datasets.items():
        command = ["-m", f"{MOD}.benchmark", "--dataset", str(dataset),
                   "--device", args.device, "--batch-size", str(args.batch_size),
                   "--output-dir", str(args.output), "--run-name", name]
        if args.smoke:
            command += ["--limit-per-direction", "20"]
        details = args.output / f"{name}_details.jsonl"
        run(command, details)
        # Deterministic scoring is cheap; rerun so it can never remain stale
        # after a benchmark details file is replaced.
        run(["-m", f"{MOD}.score_metrics", "--details", str(details),
             "--skip-comet", "--output-dir", str(args.output / "metrics")])

    if not args.skip_streaming:
        languages = ("vi", "en", "zh", "ko")
        all_directions = tuple((src, tgt) for src in languages for tgt in languages if src != tgt)
        for src, tgt in all_directions:
            output = args.output / f"streaming_{src}_{tgt}.csv"
            run(["-m", f"{MOD}.streaming_benchmark", "--dataset", str(datasets["flores"]),
                 "--src", src, "--tgt", tgt, "--chunk-tokens", "2", "4", "8",
                 "--limit", str(args.stream_limit), "--device", args.device,
                 "--output", str(output)], output)
        if not args.skip_alignatt:
            for src, tgt in all_directions:
                output = args.output / f"alignatt_{src}_{tgt}.csv"
                run(["-m", f"{MOD}.alignatt_benchmark", "--dataset", str(datasets["flores"]),
                     "--src", src, "--tgt", tgt, "--f", "1", "2", "4",
                     "--read-chunk", "4", "--limit", str(args.alignatt_limit),
                     "--device", args.device, "--output", str(output)], output)
    print(f"COMPLETE: {args.output}")


if __name__ == "__main__":
    main()
