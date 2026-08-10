"""Reference-free safety checks for live OneVoice translations."""
from __future__ import annotations

import re
from collections import Counter

NUMBER_RE = re.compile(r"(?<![\d.,:])[+-]?\d+(?:[.,:]\d+)*(?:\s*%)?(?![\d.,:])")
NEGATION = {
    "vi": (r"\bkhông\b", r"\bchưa\b", r"\bđừng\b", r"\bcấm\b"),
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
UNITS = {
    "percent": (r"%", r"\bpercent\b", r"phần trăm", r"百分比", r"퍼센트"),
    "celsius": (r"°\s*c\b", r"℃", r"\bcelsius\b", r"độ\s*c\b", r"摄氏", r"도\s*c\b"),
    "bar": (r"\bbar\b", r"(?<=\d)\s*巴", r"(?<=\d)\s*바\b"),
    "millimeter": (r"(?<![a-z])mm(?![a-z])", r"millimet(?:er|re)", r"milimét", r"毫米", r"밀리미터"),
    "volt": (r"\bvolts?\b", r"vôn", r"(?<=\d)\s*伏", r"(?<=\d)\s*볼트"),
    "decibel": (r"\bdb\b", r"\bdecibels?\b", r"đề-xi-ben", r"分贝", r"데시벨"),
}


def numbers(text: str) -> Counter:
    return Counter(re.sub(r"\s+", "", item).replace(",", ".")
                   for item in NUMBER_RE.findall(text.lower()))


def has_negation(text: str, lang: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in NEGATION[lang])


def colors(text: str, lang: str) -> set[str]:
    lowered = text.lower()
    return {concept for concept, forms in COLORS.items()
            if any(form in lowered for form in forms[lang])}


def units(text: str) -> set[str]:
    lowered = text.lower()
    return {concept for concept, patterns in UNITS.items()
            if any(re.search(pattern, lowered) for pattern in patterns)}


def check_translation(source: str, translation: str, src_lang: str, tgt_lang: str) -> dict:
    checks = {
        "numbers_preserved": numbers(source) == numbers(translation),
        "units_preserved": units(source) <= units(translation),
        "negation_preserved": not has_negation(source, src_lang) or has_negation(translation, tgt_lang),
        "colors_preserved": colors(source, src_lang) <= colors(translation, tgt_lang),
    }
    warnings = [name for name, passed in checks.items() if not passed]
    return {**checks, "safe": not warnings, "warnings": warnings,
            "source_numbers": dict(numbers(source)), "target_numbers": dict(numbers(translation)),
            "source_units": sorted(units(source)), "target_units": sorted(units(translation)),
            "source_colors": sorted(colors(source, src_lang)), "target_colors": sorted(colors(translation, tgt_lang))}
