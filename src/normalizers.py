"""
Normalization strategies for reducing hallucination in LLMs.
Two strategies implemented:
1. Text normalization (whitespace, unicode, casing, punctuation)
2. Numerical and date normalization (canonicalize numbers and dates)
"""

import unicodedata
import re
from dateutil import parser as dateparser


def normalize_text(s: str) -> str:
    """Basic text normalization: unicode NFKC, lowercase, collapse whitespace."""
    if not isinstance(s, str):
        return s
    # Unicode normalization (handles weird quotes, accents, etc.)
    s = unicodedata.normalize("NFKC", s)
    # Lowercase and strip
    s = s.lower().strip()
    # Collapse multiple whitespaces into one
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_numbers_and_dates(s: str) -> str:
    """Canonicalize numbers (word -> digit) and dates (free-form -> YYYY-MM-DD)."""
    if not isinstance(s, str):
        return s

    # Word numbers to digits (small set covering common cases)
    word_to_num = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12",
    }
    for word, num in word_to_num.items():
        s = re.sub(rf"\b{word}\b", num, s, flags=re.IGNORECASE)

    # Percent forms: "5 percent" -> "5%"
    s = re.sub(r"(\d+(?:\.\d+)?)\s*percent", r"\1%", s, flags=re.IGNORECASE)

    # Try to canonicalize common date patterns to YYYY-MM-DD
    def date_sub(match):
        try:
            parsed = dateparser.parse(match.group(0), fuzzy=False)
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            return match.group(0)

    # Match patterns like "January 5, 2024" or "Jan 5 2024"
    date_pattern = (
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\s+\d{1,2},?\s+\d{4}\b"
    )
    s = re.sub(date_pattern, date_sub, s, flags=re.IGNORECASE)

    return s


def apply_all_normalizations(s: str) -> str:
    """Apply both normalization strategies in sequence."""
    s = normalize_text(s)
    s = normalize_numbers_and_dates(s)
    return s


# Quick self-test if you run this file directly
if __name__ == "__main__":
    test_cases = [
        "  Hello   WORLD!  ",
        "I have five apples and 25 percent more.",
        "The event was on January 5, 2024.",
        "FIVE percent of January 5, 2024 transactions failed.",
    ]
    for t in test_cases:
        print(f"Original:    {t!r}")
        print(f"Text norm:   {normalize_text(t)!r}")
        print(f"Num/date:    {normalize_numbers_and_dates(t)!r}")
        print(f"Both:        {apply_all_normalizations(t)!r}")
        print()