"""Build a Kaggle-safe ZIP whose member names always use forward slashes."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path,
                   default=ROOT / "artifacts/step2/OneVoice_MT_FINAL_QUALITY.zip")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mt_out = ROOT / "outputs" / "mt"
    selected = [ROOT / "step2.md", ROOT / "STEP2_MT_FINAL_REPORT.md",
                ROOT / "requirements.txt", ROOT / "src" / "step2_mt",
                ROOT / "data" / "mt" / "manifest.json",
                mt_out / "full_kaggle", mt_out / "comet", mt_out / "finetune_eval",
                mt_out / "quality_upgrade", mt_out / "local_validation"]
    selected += list((ROOT / "data" / "mt").glob("*.jsonl"))
    selected += list(mt_out.glob("*.csv")) + list(mt_out.glob("*.jsonl"))
    selected += list((mt_out / "adapters").glob("*/training_manifest.json"))
    selected += list((mt_out / "adapters").glob("*/adapter_config.json"))
    selected += [mt_out / "nllb_ct2_int8" / "onevoice_conversion.json"]
    with ZipFile(args.output, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        seen = set()
        for item in selected:
            paths = item.rglob("*") if item.is_dir() else [item]
            for path in paths:
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                name = Path("OneVoice", path.relative_to(ROOT)).as_posix()
                if name not in seen:
                    archive.write(path, name); seen.add(name)
    print(f"{args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB, {len(seen)} files)")


if __name__ == "__main__":
    main()
