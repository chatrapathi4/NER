"""
test_inference.py — Tests for Inference and Entity Reconstruction
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.inference.predictor import (
    reconstruct_entities,
    rule_based_predict,
    Entity,
)


class TestEntityReconstruction:
    """Test BIO → entity span reconstruction."""

    def test_single_entity(self):
        tokens = ["Take", "Paracetamol", "daily"]
        labels = ["O", "B-DRUG", "O"]
        entities = reconstruct_entities(tokens, labels)
        assert len(entities) == 1
        assert entities[0].text == "Paracetamol"
        assert entities[0].label == "DRUG"

    def test_multi_token_entity(self):
        tokens = ["Take", "500", "mg", "daily"]
        labels = ["O", "B-DOSAGE", "I-DOSAGE", "O"]
        entities = reconstruct_entities(tokens, labels)
        assert len(entities) == 1
        assert entities[0].text == "500 mg"
        assert entities[0].label == "DOSAGE"

    def test_multiple_entities(self):
        tokens = ["Paracetamol", "500", "mg", "twice", "daily"]
        labels = ["B-DRUG", "B-DOSAGE", "I-DOSAGE", "B-FREQUENCY", "I-FREQUENCY"]
        entities = reconstruct_entities(tokens, labels)
        assert len(entities) == 3

        drug = [e for e in entities if e.label == "DRUG"][0]
        dosage = [e for e in entities if e.label == "DOSAGE"][0]
        freq = [e for e in entities if e.label == "FREQUENCY"][0]

        assert drug.text == "Paracetamol"
        assert dosage.text == "500 mg"
        assert freq.text == "twice daily"

    def test_consecutive_b_tags(self):
        """Two B-DRUG tags in a row → two separate entities."""
        tokens = ["Paracetamol", "Ibuprofen"]
        labels = ["B-DRUG", "B-DRUG"]
        entities = reconstruct_entities(tokens, labels)
        assert len(entities) == 2

    def test_all_outside(self):
        tokens = ["Take", "the", "medication"]
        labels = ["O", "O", "O"]
        entities = reconstruct_entities(tokens, labels)
        assert len(entities) == 0

    def test_entity_at_end(self):
        tokens = ["Take", "Paracetamol"]
        labels = ["O", "B-DRUG"]
        entities = reconstruct_entities(tokens, labels)
        assert len(entities) == 1
        assert entities[0].text == "Paracetamol"

    def test_character_offsets(self):
        text = "Take Paracetamol daily"
        tokens = ["Take", "Paracetamol", "daily"]
        labels = ["O", "B-DRUG", "O"]
        entities = reconstruct_entities(tokens, labels, original_text=text)
        assert entities[0].start == 5
        assert entities[0].end == 16


class TestRuleBasedPredictor:
    """Test the rule-based fallback predictor."""

    def test_drug_extraction(self):
        entities = rule_based_predict("Take Paracetamol daily")
        drug_entities = [e for e in entities if e.label == "DRUG"]
        assert len(drug_entities) >= 1
        assert any("Paracetamol" in e.text for e in drug_entities)

    def test_dosage_extraction(self):
        entities = rule_based_predict("Take 500 mg daily")
        dosage_entities = [e for e in entities if e.label == "DOSAGE"]
        assert len(dosage_entities) >= 1

    def test_frequency_extraction(self):
        entities = rule_based_predict("Take medication twice daily")
        freq_entities = [e for e in entities if e.label == "FREQUENCY"]
        assert len(freq_entities) >= 1

    def test_duration_extraction(self):
        entities = rule_based_predict("Take medication for 5 days")
        dur_entities = [e for e in entities if e.label == "DURATION"]
        assert len(dur_entities) >= 1

    def test_route_extraction(self):
        entities = rule_based_predict("Take medication orally")
        route_entities = [e for e in entities if e.label == "ROUTE"]
        assert len(route_entities) >= 1

    def test_form_extraction(self):
        entities = rule_based_predict("Take one tablet daily")
        form_entities = [e for e in entities if e.label == "FORM"]
        assert len(form_entities) >= 1

    def test_full_prescription(self):
        entities = rule_based_predict(
            "Take Paracetamol 500 mg twice daily for 5 days"
        )
        labels = {e.label for e in entities}
        assert "DRUG" in labels
        assert "DOSAGE" in labels
        assert "FREQUENCY" in labels
        assert "DURATION" in labels

    def test_empty_input(self):
        entities = rule_based_predict("")
        assert len(entities) == 0
