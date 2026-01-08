import re
import unicodedata
from typing import Optional


# --- Multiple-choice detection ---
_CHOICE_PAREN = re.compile(
    r"""^\s*[\(\[\{]\s*([A-Za-z])\s*[\)\]\}]\s*(?:[)/:;\-]\s*)?$""", re.X
)
_CHOICE_BARE_WITH_DELIM = re.compile(r"""^\s*([A-Za-z])\s*[)/:;\-]\s*$""", re.X)
_CHOICE_SINGLE_LETTER = re.compile(r"""^\s*([A-Za-z])\s*$""", re.X)

_YES_NO = {
    "yes": "yes",
    "y": "yes",
    "true": "yes",
    "t": "yes",
    "ok": "yes",
    "okay": "yes",
    "no": "no",
    "n": "no",
    "false": "no",
    "f": "no",
}


def _collapse_spaces(s: str) -> str:
    return " ".join(s.split())


def normalize(s: Optional[str]) -> str:
    """
    String-only normalization:
    - Multiple choice at start -> 'a'/'b'/...
    - Whole-string yes/no variants -> 'yes'/'no'
    - Else: full string lowercased, spaces collapsed (punctuation preserved)
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s).strip()
    s_low = s.lower()

    # Multiple-choice (strict)
    m = (
        _CHOICE_PAREN.match(s_low)
        or _CHOICE_BARE_WITH_DELIM.match(s_low)
        or _CHOICE_SINGLE_LETTER.match(s_low)
    )
    if m:
        return m.group(1)

    # Yes/No (match whole content, tolerate surrounding non-word chars)
    whole_clean = re.sub(r"^\W+|\W+$", "", s_low)
    if whole_clean in _YES_NO:
        return _YES_NO[whole_clean]

    # Free text: keep entire content
    return _collapse_spaces(s_low)


def equal_answer(gt: Optional[str], pred: Optional[str]) -> bool:
    return normalize(gt) == normalize(pred)


# --- quick checks ---
if __name__ == "__main__":
    assert normalize("A") == "a"
    assert not equal_answer("A", "B")
    assert not equal_answer("A.", "A")
    assert not equal_answer("A. xxxxx", "A")
    assert not equal_answer("(A).", "A")
    assert not equal_answer("(A)", "(b)")
    assert equal_answer("(A)", "A")
    assert equal_answer("two leaves", "Two   Leaves")
    assert equal_answer("YES", "yes")
    assert equal_answer("false", "no")
    assert equal_answer("Linear", "linear")
    assert equal_answer("A", "(a)")
    assert equal_answer("(A)", "(a)")
    assert equal_answer("Cartoon", "cartoon")
