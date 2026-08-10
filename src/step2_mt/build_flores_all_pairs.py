"""Convert the aligned official FLORES manifest to all 12 translation directions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "mt" / "manifest.json"
OUTPUT = ROOT / "data" / "mt" / "flores_all_pairs.jsonl"
LANGUAGES = ("vi", "en", "zh", "ko")


def main() -> None:
    aligned = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(aligned) != 1012:
        raise RuntimeError(
            f"Expected the full FLORES devtest split (1012 rows), found {len(aligned)}. "
            "Run fetch_mt_data.py again; do not report this as a full benchmark."
        )
    rows = []
    for index, item in enumerate(aligned):
        for src in LANGUAGES:
            for tgt in LANGUAGES:
                if src != tgt:
                    rows.append({"id": f"flores-{index:04d}-{src}-{tgt}", "dataset": "flores-200-devtest",
                                 "src_lang": src, "tgt_lang": tgt,
                                 "source": item[src], "reference": item[tgt]})
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} rows ({len(aligned)} sentences x 12 directions) -> {OUTPUT}")


if __name__ == "__main__":
    main()
