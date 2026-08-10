"""One-command runner for official MT datasets on Kaggle or a GPU workstation."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEP = Path(__file__).resolve().parent
DATASETS = {
    "flores": ("fetch_mt_data.py", "build_flores_all_pairs.py", ROOT / "data/mt/flores_all_pairs.jsonl"),
    "ntrex": ("fetch_ntrex.py", None, ROOT / "data/mt/ntrex_all_pairs.jsonl"),
    "massive": ("fetch_massive.py", None, ROOT / "data/mt/massive_test_all_pairs.jsonl"),
}


def run(*arguments: str) -> None:
    print("+", sys.executable, *arguments, flush=True)
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="flores,ntrex,massive",
                        help="comma-separated: flores,ntrex,massive")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke", action="store_true",
                        help="run only 20 sentences per direction before the expensive full run")
    args = parser.parse_args()

    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = set(selected) - DATASETS.keys()
    if unknown:
        raise SystemExit(f"Unknown datasets: {sorted(unknown)}")
    for name in selected:
        fetcher, converter, dataset = DATASETS[name]
        run(str(STEP / fetcher))
        if converter:
            run(str(STEP / converter))
        command = [str(STEP / "benchmark.py"), "--dataset", str(dataset),
                   "--device", args.device, "--batch-size", str(args.batch_size)]
        if args.smoke:
            command += ["--limit-per-direction", "20"]
        run(*command)


if __name__ == "__main__":
    main()
