"""
augmentation.py — Label-Aware Data Augmentation for Medical NER

Data augmentation is critical when working with limited labeled data.
In medical NER, we must be careful that augmentation does not accidentally
change entity labels or create medically nonsensical sentences.

Strategies implemented:
    1. Drug Name Synonym Replacement — swaps drug names with known synonyms
    2. Dosage Value Variation — changes numeric dosage values
    3. Frequency Expression Paraphrasing — swaps frequency phrases

Limitations of synthetic augmentation:
    - Cannot capture real clinical language variation
    - May introduce unnatural phrasing
    - Synonym lists are manually curated and incomplete
    - Does not account for drug-specific dosage constraints
"""

import random
import copy
from typing import List, Dict, Any, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Drug Name Synonyms (brand ↔ generic)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DRUG_SYNONYMS: Dict[str, List[str]] = {
    "Paracetamol": ["Acetaminophen", "Tylenol"],
    "Acetaminophen": ["Paracetamol", "Tylenol"],
    "Amoxicillin": ["Amoxil", "Trimox"],
    "Metformin": ["Glucophage", "Fortamet"],
    "Ibuprofen": ["Advil", "Motrin"],
    "Azithromycin": ["Zithromax", "Z-Pak"],
    "Omeprazole": ["Prilosec", "Losec"],
    "Ciprofloxacin": ["Cipro"],
    "Lisinopril": ["Zestril", "Prinivil"],
}

# Frequency expression alternatives
FREQUENCY_VARIANTS: Dict[str, List[str]] = {
    "twice daily": ["two times a day", "bid", "2 times daily"],
    "once daily": ["once a day", "qd", "every day"],
    "three times daily": ["three times a day", "tid", "3 times daily"],
    "every 6 hours": ["q6h", "every six hours", "6 hourly"],
    "every 8 hours": ["q8h", "every eight hours", "8 hourly"],
    "every 4 hours": ["q4h", "every four hours"],
    "every 12 hours": ["q12h", "every twelve hours", "twice daily"],
    "four times daily": ["four times a day", "qid", "4 times daily"],
}

# Common dosage units for variation
DOSAGE_UNITS = ["mg", "mcg", "g", "ml", "units"]


def augment_drug_synonym(
    record: Dict[str, Any],
    synonyms: Optional[Dict[str, List[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Replace drug names in a sentence with known synonyms.

    This augmentation preserves the BIO label structure because we only
    change the text content of B-DRUG/I-DRUG tokens, not the labels.

    Args:
        record:    Dict with 'tokens' and 'bio_labels'.
        synonyms:  Drug synonym dictionary.

    Returns:
        Augmented record, or None if no drug synonym was applicable.
    """
    if synonyms is None:
        synonyms = DEFAULT_DRUG_SYNONYMS

    new_record = copy.deepcopy(record)
    tokens = new_record["tokens"]
    labels = new_record["bio_labels"]
    augmented = False

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == "B-DRUG" and token in synonyms:
            # Pick a random synonym
            synonym = random.choice(synonyms[token])
            tokens[i] = synonym
            augmented = True

    if not augmented:
        return None

    # Update original_text to reflect changes
    new_record["original_text"] = " ".join(tokens)
    return new_record


def augment_dosage_value(
    record: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Vary numeric dosage values by a small random factor.

    For example, "500 mg" might become "250 mg" or "750 mg".
    This teaches the model that dosage numbers can vary.

    Only modifies the numeric part of B-DOSAGE tokens; units are preserved.

    Args:
        record: Dict with 'tokens' and 'bio_labels'.

    Returns:
        Augmented record, or None if no dosage was found.
    """
    new_record = copy.deepcopy(record)
    tokens = new_record["tokens"]
    labels = new_record["bio_labels"]
    augmented = False

    # Common dosage values for medical context
    common_values = [
        "50", "75", "100", "125", "150", "200", "250", "300",
        "400", "500", "600", "750", "1000",
    ]

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == "B-DOSAGE" and token.replace(".", "").isdigit():
            # Replace with a random common dosage value
            new_value = random.choice(common_values)
            tokens[i] = new_value
            augmented = True

    if not augmented:
        return None

    new_record["original_text"] = " ".join(tokens)
    return new_record


def augment_frequency(
    record: Dict[str, Any],
    variants: Optional[Dict[str, List[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Replace frequency expressions with alternative phrasings.

    This is more complex because frequency expressions are often multi-token
    (e.g., "twice daily" = B-FREQUENCY + I-FREQUENCY). We need to reconstruct
    the full phrase, find an alternative, and re-tokenize with correct labels.

    Args:
        record:    Dict with 'tokens' and 'bio_labels'.
        variants:  Frequency variant dictionary.

    Returns:
        Augmented record, or None if no frequency variant was applicable.
    """
    if variants is None:
        variants = FREQUENCY_VARIANTS

    tokens = record["tokens"]
    labels = record["bio_labels"]

    # Extract frequency spans
    freq_spans = []
    current_span_start = None

    for i, label in enumerate(labels):
        if label == "B-FREQUENCY":
            if current_span_start is not None:
                freq_spans.append((current_span_start, i))
            current_span_start = i
        elif label == "I-FREQUENCY":
            continue  # Part of current span
        else:
            if current_span_start is not None:
                freq_spans.append((current_span_start, i))
                current_span_start = None

    if current_span_start is not None:
        freq_spans.append((current_span_start, len(tokens)))

    if not freq_spans:
        return None

    new_record = copy.deepcopy(record)
    new_tokens = list(new_record["tokens"])
    new_labels = list(new_record["bio_labels"])

    # Process spans in reverse order so index shifts don't affect earlier spans
    for start, end in reversed(freq_spans):
        original_phrase = " ".join(tokens[start:end]).lower()

        # Find a variant
        for key, alts in variants.items():
            if original_phrase == key.lower():
                variant = random.choice(alts)
                variant_tokens = variant.split()

                # Build new labels for the variant tokens
                variant_labels = ["B-FREQUENCY"]
                variant_labels.extend(
                    ["I-FREQUENCY"] * (len(variant_tokens) - 1)
                )

                # Replace in token and label lists
                new_tokens[start:end] = variant_tokens
                new_labels[start:end] = variant_labels
                break

    new_record["tokens"] = new_tokens
    new_record["bio_labels"] = new_labels
    new_record["original_text"] = " ".join(new_tokens)
    return new_record


def augment_dataset(
    data: List[Dict[str, Any]],
    num_augments: int = 2,
    drug_synonyms: Optional[Dict[str, List[str]]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Apply all augmentation strategies to a dataset.

    Each original sentence may produce up to `num_augments` augmented copies.
    The original data is always preserved; augmented copies are appended.

    Args:
        data:           List of BIO-tagged records.
        num_augments:   Max augmented copies per sentence.
        drug_synonyms:  Optional drug synonym dictionary.
        seed:           Random seed for reproducibility.

    Returns:
        Original data + augmented copies.
    """
    random.seed(seed)
    augmented_data = list(data)  # Start with original data

    augmentation_funcs = [
        lambda r: augment_drug_synonym(r, drug_synonyms),
        augment_dosage_value,
        augment_frequency,
    ]

    for record in data:
        generated = 0
        # Try each augmentation strategy
        for aug_func in augmentation_funcs:
            if generated >= num_augments:
                break
            result = aug_func(record)
            if result is not None:
                augmented_data.append(result)
                generated += 1

    return augmented_data
