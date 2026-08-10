"""Build a 12-direction conversational MT benchmark from official MASSIVE 1.0 test data."""
from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "mt"
ARCHIVE = DATA_DIR / "amazon-massive-dataset-1.0.tar.gz"
OUTPUT = DATA_DIR / "massive_test_all_pairs.jsonl"
URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
LOCALES = {"vi": "vi-VN", "en": "en-US", "zh": "zh-CN", "ko": "ko-KR"}


def read_locale(archive: tarfile.TarFile, locale: str) -> dict[str, dict]:
    member_name = f"1.0/data/{locale}.jsonl"
    member = archive.getmember(member_name)
    handle = archive.extractfile(member)
    if handle is None:
        raise RuntimeError(f"Cannot read {member_name}")
    records = {}
    for raw_line in handle:
        row = json.loads(raw_line.decode("utf-8"))
        if row["partition"] == "test":
            records[str(row["id"])] = row
    return records


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print("Downloading official MASSIVE 1.0 archive ...")
        urllib.request.urlretrieve(URL, ARCHIVE)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        languages = {code: read_locale(archive, locale) for code, locale in LOCALES.items()}
    common_ids = set.intersection(*(set(records) for records in languages.values()))
    if len(common_ids) != 2974:
        raise RuntimeError(f"Expected 2974 aligned test IDs, found {len(common_ids)}")
    rows = []
    for item_id in sorted(common_ids, key=int):
        for src in LOCALES:
            for tgt in LOCALES:
                if src != tgt:
                    source = languages[src][item_id]
                    reference = languages[tgt][item_id]
                    rows.append({
                        "id": f"massive-{item_id}-{src}-{tgt}", "dataset": "massive-1.0-test",
                        "category": source["scenario"], "intent": source["intent"],
                        "src_lang": src, "tgt_lang": tgt,
                        "source": source["utt"], "reference": reference["utt"],
                    })
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} rows (2974 utterances x 12 directions) -> {OUTPUT}")


if __name__ == "__main__":
    main()
