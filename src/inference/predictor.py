"""
predictor.py — Unified NER Inference Pipeline

Provides a single interface for extracting medical entities from text,
supporting three backends:
    1. Transformer (BioBERT) — best accuracy, requires trained model
    2. CRF — lightweight baseline, requires trained model
    3. Rule-based — pattern matching fallback, always available

The rule-based mode ensures the API and frontend work immediately
without requiring any model training, which is useful for development
and demonstration.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.data.annotation import (
    tokenize_text,
    BIO_LABELS,
    label2id,
    id2label,
)
from src.preprocessing.preprocessor import preprocess_text

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Entity Data Class
# ──────────────────────────────────────────────────────────────────────────────

class Entity:
    """Represents a single extracted entity."""

    def __init__(self, text: str, label: str, start: int = 0, end: int = 0):
        self.text = text
        self.label = label
        self.start = start
        self.end = end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
        }

    def __repr__(self) -> str:
        return f"Entity(text='{self.text}', label='{self.label}')"


# ──────────────────────────────────────────────────────────────────────────────
# BIO Label → Entity Reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_entities(
    tokens: List[str],
    labels: List[str],
    original_text: Optional[str] = None,
) -> List[Entity]:
    """
    Reconstruct named entities from BIO-tagged token sequences.

    This function merges consecutive B-/I- tokens with the same entity type
    into a single entity span. It handles:
    - Single-token entities (just B-X)
    - Multi-token entities (B-X followed by I-X tokens)
    - Consecutive entities of different types

    Args:
        tokens:         List of token strings.
        labels:         List of BIO label strings (same length as tokens).
        original_text:  Optional original text for computing character offsets.

    Returns:
        List of Entity objects.
    """
    entities = []
    current_entity_tokens = []
    current_label = None

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            # Save previous entity if exists
            if current_entity_tokens and current_label:
                entity_text = " ".join(current_entity_tokens)
                entities.append(Entity(
                    text=entity_text,
                    label=current_label,
                ))
            # Start new entity
            current_label = label[2:]
            current_entity_tokens = [token]

        elif label.startswith("I-") and current_label:
            entity_type = label[2:]
            if entity_type == current_label:
                # Continue current entity
                current_entity_tokens.append(token)
            else:
                # Different type — save current and start new
                if current_entity_tokens:
                    entity_text = " ".join(current_entity_tokens)
                    entities.append(Entity(
                        text=entity_text,
                        label=current_label,
                    ))
                current_label = entity_type
                current_entity_tokens = [token]
        else:
            # O tag — save any ongoing entity
            if current_entity_tokens and current_label:
                entity_text = " ".join(current_entity_tokens)
                entities.append(Entity(
                    text=entity_text,
                    label=current_label,
                ))
            current_entity_tokens = []
            current_label = None

    # Don't forget the last entity
    if current_entity_tokens and current_label:
        entity_text = " ".join(current_entity_tokens)
        entities.append(Entity(
            text=entity_text,
            label=current_label,
        ))

    # Compute character offsets if original text provided
    if original_text:
        for entity in entities:
            idx = original_text.find(entity.text)
            if idx >= 0:
                entity.start = idx
                entity.end = idx + len(entity.text)

    return entities


# ──────────────────────────────────────────────────────────────────────────────
# Rule-Based Fallback Predictor
# ──────────────────────────────────────────────────────────────────────────────

# Known drug names for rule-based matching
KNOWN_DRUGS = {
    "paracetamol", "acetaminophen", "tylenol", "amoxicillin", "amoxil",
    "metformin", "glucophage", "ibuprofen", "advil", "motrin",
    "azithromycin", "zithromax", "omeprazole", "prilosec",
    "ciprofloxacin", "cipro", "lisinopril", "zestril",
    "prednisone", "prednisolone", "aspirin", "cetirizine",
    "salbutamol", "albuterol", "metoprolol", "amlodipine",
    "diclofenac", "doxycycline", "ranitidine", "atorvastatin",
    "gabapentin", "clopidogrel", "losartan", "pantoprazole",
    "tramadol", "warfarin", "fluconazole", "furosemide",
    "clindamycin", "betamethasone", "cephalexin", "sertraline",
    "montelukast", "levothyroxine", "hydrochlorothiazide",
    "naproxen", "morphine", "metronidazole", "erythromycin",
    "simvastatin", "ketoconazole", "vancomycin", "amitriptyline",
    "tamsulosin", "insulin", "hydrocortisone", "alprazolam", "xanax",
    "diazepam", "valium", "oxycodone", "oxycontin", "lorazepam", "ativan",
    "clonazepam", "klonopin", "codeine", "fentanyl", "hydrocodone",
}

# Regex patterns for rule-based extraction
DOSAGE_PATTERN = re.compile(
    r"\b(\d+\.?\d*)\s*(mg|mcg|g|ml|units?|iu|meq|puffs?|drops?)\b",
    re.IGNORECASE,
)

FREQUENCY_PATTERNS = [
    re.compile(r"\b(once daily|twice daily|three times daily|four times daily)\b", re.I),
    re.compile(r"\b(once a day|twice a day|three times a day)\b", re.I),
    re.compile(r"\b(every \d+ hours?)\b", re.I),
    re.compile(r"\b(every \d+ to \d+ hours?)\b", re.I),
    re.compile(r"\b(once at bedtime)\b", re.I),
    re.compile(r"\b(as needed|daily)\b", re.I),
    re.compile(r"\b(bid|tid|qid|qd|prn)\b", re.I),
]

DURATION_PATTERN = re.compile(
    r"\b(\d+)\s*(days?|weeks?|months?)\b", re.IGNORECASE,
)

ROUTE_WORDS = {
    "orally", "iv", "im", "sc", "sq", "topically",
    "subcutaneously", "intravenously", "intramuscularly",
    "rectally", "sublingually",
}

FORM_WORDS = {
    "tablet", "tablets", "capsule", "capsules", "cream", "gel",
    "ointment", "syrup", "solution", "injection", "inhaler",
    "drops", "eye drops", "suspension", "powder", "nebulizer solution",
    "ophthalmic ointment",
}


def rule_based_predict(text: str) -> List[Entity]:
    """
    Extract medical entities using regex patterns and word lists.

    This is a fallback for when no trained model is available.
    It provides reasonable results for well-formatted prescription text
    but will miss many real-world variations.

    Args:
        text: Input medical text.

    Returns:
        List of Entity objects.
    """
    entities = []
    text_lower = text.lower()

    # Extract drug names
    for drug in KNOWN_DRUGS:
        pattern = re.compile(r"\b" + re.escape(drug) + r"\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            entities.append(Entity(
                text=match.group(),
                label="DRUG",
                start=match.start(),
                end=match.end(),
            ))

    # Handle multi-word drugs like "Morphine sulfate", "IV push"
    multi_word_drugs = {
        "morphine sulfate": "DRUG",
        "iv push": "ROUTE",
    }
    for phrase, label in multi_word_drugs.items():
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            if label == "DRUG":
                # Remove individual word matches that are part of this phrase
                entities = [
                    e for e in entities
                    if not (e.start >= match.start() and e.end <= match.end() and e.label == "DRUG")
                ]
                entities.append(Entity(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                ))

    # Extract dosages
    for match in DOSAGE_PATTERN.finditer(text):
        entities.append(Entity(
            text=match.group(),
            label="DOSAGE",
            start=match.start(),
            end=match.end(),
        ))

    # Extract frequencies
    for pattern in FREQUENCY_PATTERNS:
        for match in pattern.finditer(text):
            entities.append(Entity(
                text=match.group(),
                label="FREQUENCY",
                start=match.start(),
                end=match.end(),
            ))

    # Extract durations
    for match in DURATION_PATTERN.finditer(text):
        entities.append(Entity(
            text=match.group(),
            label="DURATION",
            start=match.start(),
            end=match.end(),
        ))

    # Extract routes
    for route in ROUTE_WORDS:
        pattern = re.compile(r"\b" + re.escape(route) + r"\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            # Avoid double-tagging if already captured as part of another entity
            overlap = False
            for e in entities:
                if (match.start() >= e.start and match.end() <= e.end):
                    overlap = True
                    break
            if not overlap:
                entities.append(Entity(
                    text=match.group(),
                    label="ROUTE",
                    start=match.start(),
                    end=match.end(),
                ))

    # Extract forms
    for form in FORM_WORDS:
        pattern = re.compile(r"\b" + re.escape(form) + r"\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            overlap = False
            for e in entities:
                if (match.start() >= e.start and match.end() <= e.end):
                    overlap = True
                    break
            if not overlap:
                entities.append(Entity(
                    text=match.group(),
                    label="FORM",
                    start=match.start(),
                    end=match.end(),
                ))

    # Sort by position and remove duplicates
    entities.sort(key=lambda e: e.start)
    unique_entities = []
    seen_spans = set()
    for e in entities:
        span_key = (e.start, e.end, e.label)
        if span_key not in seen_spans:
            seen_spans.add(span_key)
            unique_entities.append(e)

    return unique_entities


# ──────────────────────────────────────────────────────────────────────────────
# Unified Predictor Class
# ──────────────────────────────────────────────────────────────────────────────

class MedicalNERPredictor:
    """
    Unified interface for medical NER prediction.

    Supports three backends:
        - "transformer": Fine-tuned BioBERT model
        - "crf":         Trained CRF model
        - "rule_based":  Regex/pattern matching (no training required)

    Args:
        model_type:             Which backend to use.
        transformer_model_dir:  Path to saved BioBERT model.
        crf_model_path:         Path to saved CRF model.
    """

    def __init__(
        self,
        model_type: str = "rule_based",
        transformer_model_dir: Optional[str] = None,
        crf_model_path: Optional[str] = None,
    ):
        self.model_type = model_type
        self.model = None
        self.tokenizer = None

        if model_type == "transformer":
            self._load_transformer(transformer_model_dir)
        elif model_type == "crf":
            self._load_crf(crf_model_path)
        elif model_type == "rule_based":
            logger.info("Using rule-based predictor (no trained model required)")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _load_transformer(self, model_dir: Optional[str]) -> None:
        """Load the fine-tuned transformer model."""
        if model_dir is None:
            model_dir = "models/biobert/final"

        try:
            from src.models.transformer_model import load_trained_model
            self.model, self.tokenizer = load_trained_model(model_dir)
            self.model.eval()
            logger.info(f"Transformer model loaded from: {model_dir}")
        except FileNotFoundError:
            logger.warning(
                f"Transformer model not found at {model_dir}. "
                "Falling back to rule-based predictor."
            )
            self.model_type = "rule_based"

    def _load_crf(self, model_path: Optional[str]) -> None:
        """Load the trained CRF model."""
        if model_path is None:
            model_path = "models/crf/crf_model.pkl"

        try:
            from src.models.crf_model import CRFModel
            self.model = CRFModel.load(model_path)
            logger.info(f"CRF model loaded from: {model_path}")
        except FileNotFoundError:
            logger.warning(
                f"CRF model not found at {model_path}. "
                "Falling back to rule-based predictor."
            )
            self.model_type = "rule_based"

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities from text.

        This is the main entry point for inference. It:
        1. Preprocesses the text
        2. Runs the appropriate model
        3. Reconstructs entities from predictions
        4. Returns structured results

        Args:
            text: Input medical/prescription text.

        Returns:
            List of entity dicts: {"text": str, "label": str, "start": int, "end": int}
        """
        # Preprocess
        text = preprocess_text(text)

        if not text:
            return []

        # Route to appropriate backend
        if self.model_type == "transformer":
            entities = self._predict_transformer(text)
        elif self.model_type == "crf":
            entities = self._predict_crf(text)
        else:
            entities = rule_based_predict(text)

        # Fall back to rule-based patterns if trained model found nothing
        if not entities and self.model_type != "rule_based":
            entities = rule_based_predict(text)

        return [e.to_dict() for e in entities]

    def _predict_transformer(self, text: str) -> List[Entity]:
        """Run transformer model inference."""
        import torch

        # Tokenize the input text
        tokens = text.split()
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        # Run model inference
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Map subword predictions back to word-level
        word_ids = encoding.word_ids(batch_index=0)
        pred_labels = []
        previous_word_id = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # Skip special tokens
            if word_id != previous_word_id:
                # First subword of a word → use its prediction
                pred_id = predictions[0][idx].item()
                pred_labels.append(id2label.get(pred_id, "O"))
            previous_word_id = word_id

        # Reconstruct entities
        return reconstruct_entities(tokens, pred_labels, text)

    def _predict_crf(self, text: str) -> List[Entity]:
        """Run CRF model inference."""
        from src.models.crf_model import sent_to_features

        # Tokenize
        token_spans = tokenize_text(text)
        tokens = [t[0] for t in token_spans]

        # Predict
        pred_labels = self.model.predict_single(tokens)

        # Reconstruct entities
        return reconstruct_entities(tokens, pred_labels, text)


def extract_entities(text: str, model_type: str = "rule_based") -> Dict[str, Any]:
    """
    Convenience function for one-shot entity extraction.

    Example:
        result = extract_entities("Take Paracetamol 500 mg twice daily for 5 days.")
        print(result)
        # {
        #     "text": "Take Paracetamol 500 mg twice daily for 5 days.",
        #     "entities": [
        #         {"text": "Paracetamol", "label": "DRUG", ...},
        #         {"text": "500 mg", "label": "DOSAGE", ...},
        #         ...
        #     ]
        # }

    Args:
        text:       Input text.
        model_type: "transformer", "crf", or "rule_based".

    Returns:
        Dict with "text" and "entities" keys.
    """
    predictor = MedicalNERPredictor(model_type=model_type)
    entities = predictor.extract_entities(text)
    return {"text": text, "entities": entities}
