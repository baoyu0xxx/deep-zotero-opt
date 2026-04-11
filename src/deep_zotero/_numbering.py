"""Helpers for safely parsing page and caption numbering."""
from __future__ import annotations

import re

_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_ROMAN_VALUES = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}


def parse_numeric_identifier(value: object) -> int | None:
    """Parse decimal or Roman-numeral identifiers into integers."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if _ROMAN_RE.fullmatch(text):
        total = 0
        prev = 0
        for char in reversed(text.upper()):
            current = _ROMAN_VALUES[char]
            if current < prev:
                total -= current
            else:
                total += current
                prev = current
        return total if total > 0 else None
    return None
