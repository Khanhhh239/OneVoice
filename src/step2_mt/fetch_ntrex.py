"""Download official Microsoft NTREX-128 references and build all 12 directions."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "mt" / "ntrex_raw"
OUTPUT = ROOT / "data" / "mt" / "ntrex_all_pairs.jsonl"
BASE = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128"
FILES = {
    "en": "newstest2019-src.eng.txt",
    "vi": "newstest2019-ref.vie.txt",
    "zh": "newstest2019-ref.zho-CN.txt",
    "ko": "newstest2019-ref.kor.txt",
}


def download(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    if not path.exists():
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(f"{BASE}/{name}", path)
    return path


def main() -> None:
    languages = {}
    for code, filename in FILES.items():
        languages[code] = download(filename).read_text(encoding="utf-8-sig").splitlines()
    lengths = {code: len(lines) for code, lines in languages.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"NTREX files are not aligned: {lengths}")
    rows = []
    for index in range(next(iter(lengths.values()))):
        for src in FILES:
            for tgt in FILES:
                if src != tgt:
                    rows.append({"id": f"ntrex-{index:04d}-{src}-{tgt}", "dataset": "ntrex-128",
                                 "src_lang": src, "tgt_lang": tgt,
                                 "source": languages[src][index], "reference": languages[tgt][index]})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} rows ({lengths['en']} sentences x 12 directions) -> {OUTPUT}")


if __name__ == "__main__":
    main()
