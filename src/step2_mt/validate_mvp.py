"""Fast, model-free acceptance checks for the Step-2 submission package."""
try:
    from .runtime_guard import NEGATION, check_translation
except ImportError:
    from runtime_guard import NEGATION, check_translation


def main() -> None:
    languages = tuple(NEGATION)
    assert set(languages) == {"vi", "en", "zh", "ko"}
    assert len([(a, b) for a in languages for b in languages if a != b]) == 12
    passing = check_translation(
        "Do not exceed 2.5 bar.", "Không được vượt quá 2,5 bar.", "en", "vi")
    assert passing["safe"], passing
    bad_number = check_translation(
        "Do not exceed 2.5 bar.", "Không được vượt quá 3 bar.", "en", "vi")
    assert not bad_number["safe"] and "numbers_preserved" in bad_number["warnings"]
    bad_color = check_translation(
        "The green light is on.", "파란색 표시등이 켜져 있습니다.", "en", "ko")
    assert not bad_color["safe"] and "colors_preserved" in bad_color["warnings"]
    print("MVP validation PASS: 4 languages, 12 directions, safety guard checks.")


if __name__ == "__main__":
    main()
