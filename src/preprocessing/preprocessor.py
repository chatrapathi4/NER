"""
preprocessor.py — Text Preprocessing for Medical NER

Preprocessing in medical NER must be conservative. Unlike sentiment analysis
or text classification, NER relies on exact character positions and entity
boundaries. Aggressive preprocessing (lowercasing, stemming, removing
punctuation) can destroy important information.

What we DO:
    - Normalize whitespace (collapse multiple spaces, strip edges)
    - Fix encoding issues (replace non-standard characters)
    - Normalize unicode characters

What we DON'T do (and why):
    - Lowercase: Drug names like "Paracetamol" are case-sensitive entities
    - Remove punctuation: Periods, commas can indicate sentence boundaries
    - Stemming/Lemmatization: Changes the surface form, breaking entity matching
    - Remove stopwords: Words like "for", "every" are part of FREQUENCY/DURATION
"""

import re
import unicodedata
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces/tabs/newlines into single spaces and strip edges.

    This is the safest preprocessing step — it doesn't change word content,
    just normalizes spacing that could confuse tokenization.

    Args:
        text: Input text.

    Returns:
        Text with normalized whitespace.
    """
    # Replace tabs, newlines, and multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode to NFC form and replace common problematic characters.

    Medical text from PDFs or OCR often contains non-standard characters
    like em-dashes, smart quotes, or non-breaking spaces.

    Args:
        text: Input text.

    Returns:
        Unicode-normalized text.
    """
    # Normalize to NFC (Canonical Decomposition + Canonical Composition)
    text = unicodedata.normalize("NFC", text)

    # Replace common problematic characters
    replacements = {
        "\u2013": "-",   # en-dash → hyphen
        "\u2014": "-",   # em-dash → hyphen
        "\u2018": "'",   # left single quote → apostrophe
        "\u2019": "'",   # right single quote → apostrophe
        "\u201c": '"',   # left double quote → straight quote
        "\u201d": '"',   # right double quote → straight quote
        "\u00a0": " ",   # non-breaking space → regular space
        "\u2022": "-",   # bullet point → hyphen
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def expand_common_abbreviations(
    text: str,
    expand: bool = False
) -> str:
    """
    Optionally expand common medical abbreviations.

    This is disabled by default because:
    1. The model should learn to recognize abbreviations in context
    2. Expanding abbreviations changes entity boundaries
    3. Some abbreviations are ambiguous (e.g., "bid" could be a word)

    When enabled, this is useful for preprocessing before human annotation.

    Args:
        text:   Input text.
        expand: Whether to actually perform expansion.

    Returns:
        Text with optionally expanded abbreviations.
    """
    if not expand:
        return text

    abbreviations = {
        r"\bqd\b": "once daily",
        r"\bbid\b": "twice daily",
        r"\btid\b": "three times daily",
        r"\bqid\b": "four times daily",
        r"\bq4h\b": "every 4 hours",
        r"\bq6h\b": "every 6 hours",
        r"\bq8h\b": "every 8 hours",
        r"\bq12h\b": "every 12 hours",
        r"\bPO\b": "orally",
        r"\bIV\b": "intravenously",
        r"\bIM\b": "intramuscularly",
        r"\bSC\b": "subcutaneously",
        r"\bSQ\b": "subcutaneously",
        r"\bPRN\b": "as needed",
        r"\btab\b": "tablet",
        r"\btabs\b": "tablets",
        r"\bcap\b": "capsule",
        r"\bcaps\b": "capsules",
    }

    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def preprocess_text(
    text: str,
    expand_abbreviations: bool = False
) -> str:
    """
    Apply the full preprocessing pipeline to a medical text.

    The pipeline is intentionally conservative to preserve entity boundaries
    and medically meaningful information.

    Args:
        text:                   Input medical text.
        expand_abbreviations:   Whether to expand abbreviations.

    Returns:
        Preprocessed text.
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Unicode normalization (fix encoding issues)
    text = normalize_unicode(text)

    # Step 2: Whitespace normalization
    text = normalize_whitespace(text)

    # Step 3: Optional abbreviation expansion
    text = expand_common_abbreviations(text, expand=expand_abbreviations)

    return text
