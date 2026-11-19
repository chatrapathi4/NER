"""
test_annotation.py — Tests for BIO Annotation Module

Tests the core annotation conversion logic:
    - Span → BIO conversion
    - Tokenization with offsets
    - Multi-token entity handling
    - BIO sequence validation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.data.annotation import (
    tokenize_text,
    convert_to_bio,
    validate_bio_sequence,
    label2id,
    id2label,
    BIO_LABELS,
    NUM_LABELS,
)


class TestTokenization:
    """Test the offset-tracking tokenizer."""

    def test_simple_sentence(self):
        tokens = tokenize_text("Take Paracetamol 500 mg")
        assert len(tokens) == 4
        assert tokens[0][0] == "Take"
        assert tokens[1][0] == "Paracetamol"
        assert tokens[2][0] == "500"
        assert tokens[3][0] == "mg"

    def test_punctuation_splitting(self):
        tokens = tokenize_text("twice daily.")
        # "daily." should be split into "daily" and "."
        assert len(tokens) == 3
        assert tokens[1][0] == "daily"
        assert tokens[2][0] == "."

    def test_character_offsets(self):
        tokens = tokenize_text("Take Paracetamol")
        # "Take" starts at 0, ends at 4
        assert tokens[0] == ("Take", 0, 4)
        # "Paracetamol" starts at 5, ends at 16
        assert tokens[1] == ("Paracetamol", 5, 16)

    def test_multiple_spaces(self):
        tokens = tokenize_text("Take  Paracetamol")
        assert len(tokens) == 2
        assert tokens[0][0] == "Take"
        assert tokens[1][0] == "Paracetamol"

    def test_empty_string(self):
        tokens = tokenize_text("")
        assert tokens == []


class TestBIOConversion:
    """Test span-to-BIO conversion."""

    def test_single_token_entity(self):
        text = "Take Paracetamol daily"
        entities = [
            {"start": 5, "end": 16, "label": "DRUG", "text": "Paracetamol"}
        ]
        tokens, labels = convert_to_bio(text, entities)
        assert "B-DRUG" in labels
        drug_idx = labels.index("B-DRUG")
        assert tokens[drug_idx] == "Paracetamol"

    def test_multi_token_entity(self):
        text = "Take 500 mg daily"
        entities = [
            {"start": 5, "end": 11, "label": "DOSAGE", "text": "500 mg"}
        ]
        tokens, labels = convert_to_bio(text, entities)
        assert "B-DOSAGE" in labels
        assert "I-DOSAGE" in labels

        b_idx = labels.index("B-DOSAGE")
        i_idx = labels.index("I-DOSAGE")
        assert tokens[b_idx] == "500"
        assert tokens[i_idx] == "mg"

    def test_multiple_entities(self):
        text = "Paracetamol 500 mg twice daily"
        entities = [
            {"start": 0, "end": 11, "label": "DRUG", "text": "Paracetamol"},
            {"start": 12, "end": 18, "label": "DOSAGE", "text": "500 mg"},
            {"start": 19, "end": 30, "label": "FREQUENCY", "text": "twice daily"},
        ]
        tokens, labels = convert_to_bio(text, entities)

        assert "B-DRUG" in labels
        assert "B-DOSAGE" in labels
        assert "I-DOSAGE" in labels
        assert "B-FREQUENCY" in labels
        assert "I-FREQUENCY" in labels

    def test_outside_tokens(self):
        text = "Take Paracetamol after food"
        entities = [
            {"start": 5, "end": 16, "label": "DRUG", "text": "Paracetamol"}
        ]
        tokens, labels = convert_to_bio(text, entities)

        # "Take", "after", "food" should all be O
        assert labels[0] == "O"
        assert labels[-1] == "O"

    def test_no_entities(self):
        text = "Take this medication"
        entities = []
        tokens, labels = convert_to_bio(text, entities)
        assert all(label == "O" for label in labels)


class TestBIOValidation:
    """Test BIO sequence validation/correction."""

    def test_valid_sequence(self):
        tokens = ["Take", "500", "mg"]
        labels = ["O", "B-DOSAGE", "I-DOSAGE"]
        corrected = validate_bio_sequence(tokens, labels)
        assert corrected == labels  # No changes needed

    def test_orphan_i_tag(self):
        tokens = ["Take", "mg"]
        labels = ["O", "I-DOSAGE"]
        corrected = validate_bio_sequence(tokens, labels)
        # I-DOSAGE without preceding B-DOSAGE should be corrected to B-DOSAGE
        assert corrected[1] == "B-DOSAGE"

    def test_i_tag_after_different_type(self):
        tokens = ["Paracetamol", "mg"]
        labels = ["B-DRUG", "I-DOSAGE"]
        corrected = validate_bio_sequence(tokens, labels)
        # I-DOSAGE after B-DRUG should become B-DOSAGE
        assert corrected[1] == "B-DOSAGE"

    def test_length_mismatch(self):
        tokens = ["Take"]
        labels = ["O", "B-DRUG"]
        with pytest.raises(ValueError):
            validate_bio_sequence(tokens, labels)


class TestLabelMappings:
    """Test label2id and id2label consistency."""

    def test_num_labels(self):
        assert NUM_LABELS == 13

    def test_o_label(self):
        assert label2id["O"] == 0
        assert id2label[0] == "O"

    def test_all_entity_types_present(self):
        for entity in ["DRUG", "DOSAGE", "FREQUENCY", "DURATION", "ROUTE", "FORM"]:
            assert f"B-{entity}" in label2id
            assert f"I-{entity}" in label2id

    def test_bijective(self):
        for label, idx in label2id.items():
            assert id2label[idx] == label

    def test_bio_labels_list(self):
        assert BIO_LABELS[0] == "O"
        assert "B-DRUG" in BIO_LABELS
        assert "I-DRUG" in BIO_LABELS
