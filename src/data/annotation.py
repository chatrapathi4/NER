"""
annotation.py — BIO Annotation Conversion Module

This module converts span-level entity annotations into BIO-tagged token
sequences, the standard format for training NER models.

BIO Format:
    B- (Beginning): First token of a named entity
    I- (Inside):    Subsequent tokens of a multi-token entity
    O  (Outside):   Tokens that are not part of any entity

Example:
    Text: "Take Paracetamol 500 mg twice daily."
    Tokens:  ["Take", "Paracetamol", "500", "mg", "twice", "daily", "."]
    Labels:  ["O",    "B-DRUG",      "B-DOSAGE", "I-DOSAGE", "B-FREQUENCY", "I-FREQUENCY", "O"]
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Label Configuration
# ──────────────────────────────────────────────────────────────────────────────

ENTITY_TYPES = ["DRUG", "DOSAGE", "FREQUENCY", "DURATION", "ROUTE", "FORM"]

# Build the full BIO label list: O, B-DRUG, I-DRUG, B-DOSAGE, I-DOSAGE, ...
BIO_LABELS = ["O"]
for entity_type in ENTITY_TYPES:
    BIO_LABELS.append(f"B-{entity_type}")
    BIO_LABELS.append(f"I-{entity_type}")

# Mapping from label string to integer ID and vice versa
# Used by both the CRF and transformer models
label2id: Dict[str, int] = {label: idx for idx, label in enumerate(BIO_LABELS)}
id2label: Dict[int, str] = {idx: label for idx, label in enumerate(BIO_LABELS)}

NUM_LABELS = len(BIO_LABELS)  # 13 labels total


# ──────────────────────────────────────────────────────────────────────────────
# Core Functions
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_text(text: str) -> List[Tuple[str, int, int]]:
    """
    Simple whitespace + punctuation tokenizer that tracks character offsets.

    Returns a list of (token, start_char, end_char) tuples.
    We use a simple tokenizer here so that BIO labels can be assigned based
    on character-level overlap with entity spans.

    Args:
        text: Input text string.

    Returns:
        List of tuples: (token_text, start_offset, end_offset)
    """
    tokens = []
    i = 0
    n = len(text)

    while i < n:
        # Skip whitespace
        if text[i].isspace():
            i += 1
            continue

        # Collect contiguous non-space characters, but split off
        # trailing punctuation (period, comma, semicolon, colon)
        start = i
        while i < n and not text[i].isspace():
            i += 1
        raw_token = text[start:i]

        # Split trailing punctuation from the token
        # e.g. "daily." → ["daily", "."]
        if len(raw_token) > 1 and raw_token[-1] in ".,;:!?":
            tokens.append((raw_token[:-1], start, i - 1))
            tokens.append((raw_token[-1], i - 1, i))
        else:
            tokens.append((raw_token, start, i))

    return tokens


def convert_to_bio(
    text: str,
    entities: List[Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """
    Convert a span-annotated sentence into BIO-tagged token sequences.

    This is the core annotation conversion function. For each token, we check
    if it overlaps with any entity span. If a token is the first token in an
    entity span, it gets a B- tag; subsequent tokens in the same span get I-.
    Tokens outside all entity spans get the O tag.

    Args:
        text:     The original sentence text.
        entities: List of entity dicts with keys: start, end, label, text.

    Returns:
        Tuple of (tokens, bio_labels) — parallel lists of the same length.
    """
    # Step 1: Tokenize the text with character offsets
    token_spans = tokenize_text(text)
    tokens = [t[0] for t in token_spans]
    bio_labels = ["O"] * len(tokens)

    # Step 2: Sort entities by start position for deterministic processing
    sorted_entities = sorted(entities, key=lambda e: e["start"])

    # Step 3: For each token, check if it falls within any entity span
    for idx, (token_text, tok_start, tok_end) in enumerate(token_spans):
        for entity in sorted_entities:
            ent_start = entity["start"]
            ent_end = entity["end"]
            ent_label = entity["label"]

            # Check if token overlaps with entity span
            # A token overlaps if it starts within the entity span
            if tok_start >= ent_start and tok_start < ent_end:
                # Determine if this is the first token of the entity (B-)
                # or a continuation token (I-)
                if tok_start == ent_start:
                    bio_labels[idx] = f"B-{ent_label}"
                else:
                    bio_labels[idx] = f"I-{ent_label}"
                break  # Token matched; no need to check other entities

    return tokens, bio_labels


def load_annotations(filepath: str) -> List[Dict[str, Any]]:
    """
    Load span-annotated sentences from a JSONL file.

    Each line in the file should be a JSON object with:
        - "text": the sentence string
        - "entities": list of {"start": int, "end": int, "label": str, "text": str}

    Args:
        filepath: Path to the JSONL file.

    Returns:
        List of annotation dictionaries.
    """
    annotations = []
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Annotation file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Validate required fields
                if "text" not in record or "entities" not in record:
                    raise ValueError(
                        f"Line {line_num}: Missing 'text' or 'entities' field."
                    )
                annotations.append(record)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Line {line_num}: Invalid JSON — {e}"
                ) from e

    return annotations


def convert_annotations_to_bio(
    annotations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert a list of span-annotated sentences to BIO-tagged format.

    Args:
        annotations: List of dicts with 'text' and 'entities'.

    Returns:
        List of dicts with 'tokens' and 'bio_labels'.
    """
    bio_data = []

    for record in annotations:
        tokens, bio_labels = convert_to_bio(
            record["text"], record["entities"]
        )
        bio_data.append({
            "tokens": tokens,
            "bio_labels": bio_labels,
            "original_text": record["text"],
        })

    return bio_data


def save_bio_data(bio_data: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save BIO-tagged data to a JSONL file.

    Args:
        bio_data:  List of dicts with 'tokens' and 'bio_labels'.
        filepath:  Output file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for record in bio_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_bio_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load BIO-tagged data from a JSONL file.

    Args:
        filepath: Path to the JSONL file.

    Returns:
        List of dicts with 'tokens' and 'bio_labels'.
    """
    data = []
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"BIO data file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def validate_bio_sequence(tokens: List[str], labels: List[str]) -> List[str]:
    """
    Validate and fix common BIO tagging errors.

    Rules:
        - I-X cannot follow O or B-Y (where Y != X)
        - If an I-X tag appears without a preceding B-X, convert it to B-X

    Args:
        tokens: List of token strings.
        labels: List of BIO label strings.

    Returns:
        Corrected list of BIO labels.
    """
    if len(tokens) != len(labels):
        raise ValueError(
            f"Token count ({len(tokens)}) != label count ({len(labels)})"
        )

    corrected = list(labels)

    for i, label in enumerate(corrected):
        if label.startswith("I-"):
            entity_type = label[2:]
            # Check previous label
            if i == 0:
                # I- at the start → must be B-
                corrected[i] = f"B-{entity_type}"
            else:
                prev = corrected[i - 1]
                # Previous must be B-X or I-X with same entity type
                if prev not in (f"B-{entity_type}", f"I-{entity_type}"):
                    corrected[i] = f"B-{entity_type}"

    return corrected
