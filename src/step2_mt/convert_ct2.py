"""Convert NLLB to a deployable CTranslate2 int8 model and record its size."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quantization", choices=["int8", "int8_float16", "float16"], default="int8")
    args = p.parse_args()
    try:
        import ctranslate2
    except ImportError as exc:
        raise SystemExit("Install ctranslate2 first: pip install ctranslate2") from exc
    ctranslate2.converters.TransformersConverter(args.model).convert(
        str(args.output), quantization=args.quantization, force=True)
    size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
    manifest = {"source_model": args.model, "quantization": args.quantization,
                "bytes": size, "mib": round(size / 1024**2, 2)}
    (args.output / "onevoice_conversion.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
