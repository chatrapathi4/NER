"""
transformer_model.py — BioBERT Transformer Model for Medical NER

This module wraps the HuggingFace Transformers library for fine-tuning
a pretrained biomedical transformer (BioBERT) on our medical NER task.

Why BioBERT (dmis-lab/biobert-base-cased-v1.2)?
    1. Pre-trained on PubMed abstracts (~4.5B words) + PMC full-text articles
    2. Better understanding of biomedical terminology vs. general BERT
    3. Cased model preserves capitalization (important for drug names)
    4. Well-supported on HuggingFace Hub
    5. Compatible with standard BERT architecture (easy to fine-tune)

Architecture:
    BioBERT base → [CLS] tok1 tok2 ... [SEP] [PAD] ...
                    ↓     ↓    ↓   ↓    ↓     ↓
                   drop  drop drop drop drop  drop
                    ↓     ↓    ↓   ↓    ↓     ↓
                   FC    FC   FC  FC   FC    FC    (Linear: 768 → 13)
                    ↓     ↓    ↓   ↓    ↓     ↓
                  logits for each of the 13 BIO labels

    Only the token-level outputs are used (not [CLS]).
    Each token gets a probability distribution over 13 BIO labels.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
)

from src.data.annotation import label2id, id2label, NUM_LABELS


# ──────────────────────────────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Default model name — can be overridden via config.yaml
DEFAULT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"


def load_tokenizer(model_name: Optional[str] = None) -> AutoTokenizer:
    """
    Load the tokenizer corresponding to the pretrained model.

    Why AutoTokenizer?
        - Automatically selects the right tokenizer class for the model
        - Ensures the tokenizer matches the model's vocabulary exactly
        - from_pretrained() downloads and caches the tokenizer files

    Why must the tokenizer match the model?
        The tokenizer defines the vocabulary (word → ID mapping) that the
        model was trained with. Using a different tokenizer would produce
        incorrect token IDs, leading to garbage predictions.

    Args:
        model_name: HuggingFace model identifier or local path.

    Returns:
        Configured AutoTokenizer instance.
    """
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # use_fast=True enables the Rust-based tokenizer which is
        # significantly faster and supports word_ids() for label alignment
        use_fast=True,
    )
    return tokenizer


def load_model(
    model_name: Optional[str] = None,
    num_labels: int = NUM_LABELS,
) -> AutoModelForTokenClassification:
    """
    Load a pretrained transformer for token classification.

    Why AutoModelForTokenClassification?
        This class adds a token classification head (linear layer) on top of
        the base transformer. Each token in the sequence gets mapped from
        the hidden dimension (768 for BERT-base) to `num_labels` logits.

    Why from_pretrained()?
        - Downloads the pretrained weights from HuggingFace Hub
        - The base transformer layers are initialized with weights learned
          from biomedical text (transfer learning)
        - The classification head is randomly initialized and will be
          fine-tuned on our NER data

    What is num_labels?
        The number of output classes. For our BIO schema, this is 13:
        O, B-DRUG, I-DRUG, B-DOSAGE, I-DOSAGE, B-FREQUENCY, I-FREQUENCY,
        B-DURATION, I-DURATION, B-ROUTE, I-ROUTE, B-FORM, I-FORM.

    Args:
        model_name: HuggingFace model identifier or local path.
        num_labels: Number of NER labels.

    Returns:
        Configured model ready for fine-tuning.
    """
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    # Load the model configuration and update it with our label mappings
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
        # ignore_mismatched_sizes=True allows loading a pretrained model
        # even if the classification head has a different number of labels
        # (the pretrained model may have been trained for a different task)
        ignore_mismatched_sizes=True,
    )

    return model


def load_trained_model(
    model_dir: str,
) -> tuple:
    """
    Load a fine-tuned model and tokenizer from a local directory.

    After training, the model and tokenizer are saved to disk.
    This function loads them for inference.

    Args:
        model_dir: Path to the directory containing the saved model.

    Returns:
        Tuple of (model, tokenizer).
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            "Please train the model first with: python scripts/train.py --model transformer"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_path))

    return model, tokenizer
