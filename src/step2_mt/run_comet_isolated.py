"""Run COMET without replacing Kaggle's working Torch/CUDA installation."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STACK_VERSION = "comet-2.2.7-transformers-4.44.2-v1"
PACKAGES = [
    "numpy==1.26.4", "protobuf==4.25.9", "jsonargparse==3.13.1",
    "entmax==1.3", "torchmetrics==0.10.3", "lightning-utilities==0.15.3",
    "pytorch-lightning==2.6.5", "unbabel-comet==2.2.7",
    "transformers==4.44.2", "tokenizers==0.19.1", "huggingface-hub==0.36.2",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--details", type=Path, required=True)
    p.add_argument("--env-dir", type=Path, default=ROOT / ".comet_packages_clean")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mt" / "comet")
    p.add_argument("--model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gpus", type=int, default=1)
    args = p.parse_args()
    if not args.details.exists():
        raise FileNotFoundError(args.details)

    args.env_dir.mkdir(parents=True, exist_ok=True)
    marker = args.env_dir / f".{STACK_VERSION}"
    runtime = Path("/usr/local/bin/python") if Path("/usr/local/bin/python").exists() else Path(sys.executable)
    if not marker.exists():
        subprocess.run([str(runtime), "-m", "pip", "install", "--no-deps",
                        "--upgrade", "--target", str(args.env_dir), *PACKAGES], check=True)
        marker.write_text("installed\n", encoding="utf-8")

    old_pythonpath = os.environ.get("PYTHONPATH", "")
    env = dict(os.environ, PYTHONNOUSERSITE="1",
               PYTHONPATH=str(args.env_dir) + (os.pathsep + old_pythonpath if old_pythonpath else ""))
    check = ("import torch,torchvision,transformers,huggingface_hub,comet;"
             "print(torch.__version__,torchvision.__version__,transformers.__version__,"
             "huggingface_hub.__version__)")
    subprocess.run([str(runtime), "-c", check], cwd=ROOT, env=env, check=True)
    subprocess.run([str(runtime), "-m", "src.step2_mt.score_metrics",
                    "--details", str(args.details.resolve()),
                    "--output-dir", str(args.output_dir.resolve()),
                    "--comet-model", args.model, "--batch-size", str(args.batch_size),
                    "--gpus", str(args.gpus)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
