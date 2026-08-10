"""Download gated PhoMT.zip and export local, non-redistributable JSONL."""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from itertools import zip_longest
from pathlib import Path


def classify(path: Path) -> tuple[str | None, str | None]:
    name, full = path.name.lower(), str(path).lower()
    lang = "vi" if any(x in name for x in (".vi", ".vie", "_vi", "-vi", "vietnamese")) else None
    if any(x in name for x in (".en", ".eng", "_en", "-en", "english")):
        lang = "en"
    split = "train" if "train" in full else None
    if any(x in full for x in ("validation", "valid", "dev")):
        split = "validation"
    elif "test" in full:
        split = "test"
    return split, lang


def convert(split: str, vi_path: Path, en_path: Path, output: Path, limit: int | None) -> int:
    count = 0
    with vi_path.open(encoding="utf-8-sig") as vi, en_path.open(encoding="utf-8-sig") as en, \
            output.open("w", encoding="utf-8") as target:
        for index, pair in enumerate(zip_longest(vi, en)):
            vi_line, en_line = pair
            if vi_line is None or en_line is None:
                raise ValueError(f"PhoMT {split} files differ in length at line {index}")
            vi_text, en_text = vi_line.strip(), en_line.strip()
            if not vi_text or not en_text:
                continue
            for src, tgt, source, reference in (("vi", "en", vi_text, en_text),
                                                 ("en", "vi", en_text, vi_text)):
                target.write(json.dumps({"id": f"phomt-{split}-{count}-{src}-{tgt}",
                    "src_lang": src, "tgt_lang": tgt, "source": source,
                    "reference": reference}, ensure_ascii=False) + "\n")
            count += 1
            if limit and count >= limit:
                break
    return count


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--accept-phomt-terms", action="store_true")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if not args.accept_phomt_terms:
        raise SystemExit("Accept VinAI PhoMT research/education terms first")
    from huggingface_hub import get_token, hf_hub_download
    token = get_token() or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Authenticate with Hugging Face before downloading gated PhoMT")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = Path(hf_hub_download("vinai/PhoMT", "PhoMT.zip", repo_type="dataset",
                                       token=token, local_dir=args.output_dir))
    raw = args.output_dir / "raw"
    raw.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (raw / member.filename).resolve()
            if not str(destination).startswith(str(raw.resolve())):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(raw)
    detected: dict[str, dict[str, Path]] = {}
    for file in raw.rglob("*"):
        if file.is_file():
            split, lang = classify(file)
            if split and lang:
                detected.setdefault(split, {})[lang] = file
    for split, paths in detected.items():
        if {"vi", "en"} <= paths.keys():
            output = args.output_dir / f"phomt_{split}.jsonl"
            print(split, convert(split, paths["vi"], paths["en"], output, args.limit), output)


if __name__ == "__main__":
    main()
