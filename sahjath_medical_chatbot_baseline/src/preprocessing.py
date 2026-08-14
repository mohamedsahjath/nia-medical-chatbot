import re
from functools import lru_cache
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "should", "the", "to", "what", "when", "with", "you", "your",
}


@lru_cache(maxsize=2048)
def preprocess_text(text: str) -> str:
    """Normalize, tokenize, remove light stop words, and reduce common suffixes."""
    text = str(text).lower().strip()
    tokens = re.findall(r"[a-z]+", text)
    useful = (_normalize_word(token) for token in tokens if token not in _STOP_WORDS)
    return " ".join(useful)


def _normalize_word(word: str) -> str:
    """Small dependency-free normalizer suitable for this transparent prototype."""
    for suffix in ("ingly", "edly", "ing", "ed", "ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word
