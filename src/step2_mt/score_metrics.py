"""Add neural and deterministic safety metrics to an existing benchmark output.

Translation inference is deliberately separate: if COMET runs out of memory,
the expensive translations remain on disk and can be scored again.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

NUMBER_RE = re.compile(r"(?<![\d.,:])[+-]?\d+(?:[.,:]\d+)*(?:\s*%)?(?![\d.,:])")
LATIN_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9._+-]*\b")

NEGATION = {
    "vi": (r"\bkhông\b", r"\bchưa\b", r"\bđừng\b", r"\bchẳng\b", r"\bcấm\b"),
    "en": (r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bdo(?:n't| not)\b", r"\bmust(?:n't| not)\b"),
    "zh": ("不", "未", "无", "勿", "禁止", "不得", "没有"),
    "ko": ("않", "안 ", "못", "금지", "마십시오", "없"),
}
COLORS = {
    "red": {"vi": ("đỏ",), "en": ("red",), "zh": ("红",), "ko": ("빨간", "적색")},
    "green": {"vi": ("xanh lá", "màu xanh"), "en": ("green",), "zh": ("绿",), "ko": ("초록", "녹색")},
    "blue": {"vi": ("xanh dương",), "en": ("blue",), "zh": ("蓝",), "ko": ("파란", "청색")},
    "yellow": {"vi": ("vàng",), "en": ("yellow",), "zh": ("黄",), "ko": ("노란", "황색")},
}
UNIT_PATTERNS = {
    "percent": (r"%", r"\bpercent\b", r"phần trăm", r"百分比", r"퍼센트"),
    "celsius": (r"°\s*c\b", r"℃", r"\bcelsius\b", r"độ\s*c\b", r"摄氏", r"도\s*c\b"),
    "bar": (r"\bbar\b", r"(?<=\d)\s*巴", r"(?<=\d)\s*바\b"),
    "millimeter": (r"(?<![a-z])mm(?![a-z])", r"\bmillimet(?:er|re)s?\b", r"milimét", r"毫米", r"밀리미터"),
    "volt": (r"\bvolts?\b", r"vôn", r"(?<=\d)\s*伏", r"(?<=\d)\s*볼트"),
    "decibel": (r"\bdb\b", r"\bdecibels?\b", r"đề-xi-ben", r"分贝", r"데시벨"),
}


def normalized_numbers(text: str) -> Counter:
    values = []
    for match in NUMBER_RE.findall(text.lower()):
        value = re.sub(r"\s+", "", match).replace(",", ".")
        values.append(value)
    return Counter(values)


def concepts(text: str, vocabulary: dict[str, tuple[str, ...]]) -> set[str]:
    lowered = text.lower()
    return {name for name, forms in vocabulary.items() if any(form.lower() in lowered for form in forms)}


def unit_concepts(text: str) -> set[str]:
    lowered = text.lower()
    return {name for name, patterns in UNIT_PATTERNS.items()
            if any(re.search(pattern, lowered) for pattern in patterns)}


def has_negation(text: str, language: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in NEGATION[language])


def color_concepts(text: str, language: str) -> set[str]:
    lowered = text.lower()
    return {name for name, by_language in COLORS.items()
            if any(form in lowered for form in by_language[language])}


def latin_anchors(text: str) -> set[str]:
    return {re.sub(r"[._+-]", "", token.casefold()) for token in LATIN_TOKEN_RE.findall(text)
            if any(char.isdigit() for char in token) or token.isupper()}


def deterministic_metrics(row: dict) -> dict:
    reference, hypothesis, language = row["reference"], row["hypothesis"], row["tgt_lang"]
    expected_numbers, actual_numbers = normalized_numbers(reference), normalized_numbers(hypothesis)
    expected_units, actual_units = unit_concepts(reference), unit_concepts(hypothesis)
    expected_colors, actual_colors = color_concepts(reference, language), color_concepts(hypothesis, language)
    expected_anchors, actual_anchors = latin_anchors(reference), latin_anchors(hypothesis)
    # If the human reference spells every number as words, exact numeric
    # preservation is not measurable without language-specific number-word
    # normalization. Do not penalize a valid digit rendering in that case.
    number_pass = not expected_numbers or expected_numbers == actual_numbers
    unit_pass = expected_units <= actual_units
    negation_pass = not has_negation(reference, language) or has_negation(hypothesis, language)
    color_pass = expected_colors <= actual_colors and not bool(actual_colors - expected_colors)
    anchor_pass = expected_anchors <= actual_anchors
    reference_length = max(len(reference.strip()), 1)
    return {
        "numbers_expected": dict(expected_numbers), "numbers_actual": dict(actual_numbers),
        "number_pass": number_pass, "units_expected": sorted(expected_units),
        "units_actual": sorted(actual_units), "unit_pass": unit_pass,
        "negation_expected": has_negation(reference, language), "negation_pass": negation_pass,
        "colors_expected": sorted(expected_colors), "colors_actual": sorted(actual_colors),
        "color_pass": color_pass, "anchors_expected": sorted(expected_anchors),
        "anchors_actual": sorted(actual_anchors), "anchor_pass": anchor_pass,
        "length_ratio": round(len(hypothesis.strip()) / reference_length, 4),
        "safety_pass": number_pass and unit_pass and negation_pass and color_pass and anchor_pass,
    }


def run_comet(rows: list[dict], model_name: str, batch_size: int, gpus: int) -> tuple[list[float], object]:
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as error:
        raise SystemExit("Install neural metrics first: pip install unbabel-comet") from error
    checkpoint = download_model(model_name)
    model = load_from_checkpoint(checkpoint)
    samples = [{"src": row["source"], "mt": row["hypothesis"], "ref": row["reference"]}
               for row in rows]
    output = model.predict(samples, batch_size=batch_size, gpus=gpus)
    scores = [float(value) for value in output.scores]
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return scores, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--xcomet-model", help="Optional gated model, e.g. Unbabel/XCOMET-XL")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--skip-comet", action="store_true")
    args = parser.parse_args()

    with args.details.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        row.update(deterministic_metrics(row))

    if not args.skip_comet:
        scores, _ = run_comet(rows, args.comet_model, args.batch_size, args.gpus)
        for row, score in zip(rows, scores):
            row["comet_model"] = args.comet_model
            row["comet_score"] = score

    if args.xcomet_model:
        scores, output = run_comet(rows, args.xcomet_model, args.batch_size, args.gpus)
        error_spans = getattr(getattr(output, "metadata", None), "error_spans", None)
        for index, (row, score) in enumerate(zip(rows, scores)):
            spans = error_spans[index] if error_spans is not None else []
            row["xcomet_model"] = args.xcomet_model
            row["xcomet_score"] = score
            row["xcomet_error_spans"] = spans
            severities = Counter(span.get("severity", "unknown") for span in spans)
            row["xcomet_minor_errors"] = severities["minor"]
            row["xcomet_major_errors"] = severities["major"]
            row["xcomet_critical_errors"] = severities["critical"]

    output_dir = args.output_dir or args.details.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.details.stem.removesuffix("_details")
    enriched_path = output_dir / f"{stem}_metrics_details.jsonl"
    enriched_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    groups = defaultdict(list)
    for row in rows:
        groups[f'{row["src_lang"]}->{row["tgt_lang"]}'].append(row)
    summaries = []
    for direction, part in sorted(groups.items()):
        summary = {"direction": direction, "n": len(part)}
        for field in ("number_pass", "unit_pass", "negation_pass", "color_pass", "anchor_pass", "safety_pass"):
            summary[f"{field}_rate"] = round(sum(bool(row[field]) for row in part) / len(part), 4)
        summary["length_ratio_mean"] = round(sum(row["length_ratio"] for row in part) / len(part), 4)
        if "comet_score" in part[0]:
            summary["comet_mean"] = round(sum(row["comet_score"] for row in part) / len(part), 4)
        if "xcomet_score" in part[0]:
            summary["xcomet_mean"] = round(sum(row["xcomet_score"] for row in part) / len(part), 4)
            for severity in ("minor", "major", "critical"):
                field = f"xcomet_{severity}_errors"
                summary[field] = sum(row[field] for row in part)
        summaries.append(summary)

    summary_path = output_dir / f"{stem}_metrics_summary.csv"
    fields = list(dict.fromkeys(key for row in summaries for key in row))
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"details={enriched_path}\nsummary={summary_path}")


if __name__ == "__main__":
    main()
