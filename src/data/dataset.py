"""
dataset.py — HuggingFace Dataset Creation for Transformer Training

This module converts BIO-tagged data into a format compatible with the
HuggingFace Transformers library, handling the critical step of aligning
BIO labels with subword tokens produced by the transformer's tokenizer.

Why label alignment matters:
    Transformers use subword tokenization (WordPiece/BPE). A word like
    "Azithromycin" might be split into ["Az", "##ith", "##rom", "##ycin"].
    We need to decide what label each subword gets:
    - First subword: inherits the word's label (e.g., B-DRUG)
    - Subsequent subwords: get the same I- label (continuation)
    - Special tokens ([CLS], [SEP], [PAD]): get label -100
      (ignored by PyTorch's CrossEntropyLoss during training)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from src.data.annotation import label2id, id2label, BIO_LABELS


class MedicalNERDataset(Dataset):
    """
    PyTorch Dataset for Medical NER token classification.

    Handles:
        1. Loading BIO-tagged data from JSONL files
        2. Tokenizing with a HuggingFace tokenizer
        3. Aligning BIO labels with subword tokens
        4. Returning tensors ready for the Trainer API

    Args:
        data:       List of dicts with 'tokens' and 'bio_labels'.
        tokenizer:  HuggingFace tokenizer instance.
        max_length: Maximum sequence length (pads/truncates to this).
    """

    # Label assigned to special tokens and non-first subwords of a word.
    # PyTorch CrossEntropyLoss ignores targets with this value.
    IGNORE_LABEL_ID = -100

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 128
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Tokenize a single example and align labels with subword tokens.

        Returns a dict with:
            - input_ids:      Token IDs for the transformer
            - attention_mask:  1 for real tokens, 0 for padding
            - labels:          Aligned BIO label IDs (-100 for specials)
        """
        record = self.data[idx]
        tokens = record["tokens"]
        bio_labels = record["bio_labels"]

        # ── Step 1: Tokenize with is_split_into_words=True ──
        # This tells the tokenizer that our input is already pre-tokenized
        # (split into words). The tokenizer will then apply subword
        # tokenization to each word individually.
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ── Step 2: Align labels with subword tokens ──
        # word_ids() maps each subword token back to its original word index.
        # None means the token is a special token ([CLS], [SEP], [PAD]).
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special token → ignore in loss
                aligned_labels.append(self.IGNORE_LABEL_ID)
            elif word_id != previous_word_id:
                # First subword of a new word → use the word's label
                label_str = bio_labels[word_id]
                aligned_labels.append(label2id.get(label_str, 0))
            else:
                # Continuation subword of the same word
                # If the word's label is B-X, the subword gets I-X
                # If the word's label is I-X or O, the subword gets the same
                label_str = bio_labels[word_id]
                if label_str.startswith("B-"):
                    # Convert B-DRUG → I-DRUG for subword continuations
                    entity_type = label_str[2:]
                    cont_label = f"I-{entity_type}"
                    aligned_labels.append(label2id.get(cont_label, 0))
                elif label_str == "O":
                    aligned_labels.append(label2id["O"])
                else:
                    # Already I-X, keep as is
                    aligned_labels.append(label2id.get(label_str, 0))

            previous_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def load_dataset_from_jsonl(
    filepath: str,
    tokenizer=None,
    max_length: int = 128
) -> MedicalNERDataset:
    """
    Load a JSONL file and create a MedicalNERDataset.

    Args:
        filepath:   Path to the JSONL file.
        tokenizer:  HuggingFace tokenizer (required for transformer training).
        max_length: Maximum sequence length.

    Returns:
        MedicalNERDataset instance.
    """
    data = []
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if tokenizer is None:
        raise ValueError("A tokenizer is required to create the dataset.")

    return MedicalNERDataset(data, tokenizer, max_length)


def create_hf_dataset(
    data: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 128
) -> "datasets.Dataset":
    """
    Create a HuggingFace datasets.Dataset from BIO data.

    This is an alternative to the PyTorch Dataset class, useful when
    working with the HuggingFace Trainer API which natively supports
    the datasets library format.

    Args:
        data:       List of dicts with 'tokens' and 'bio_labels'.
        tokenizer:  HuggingFace tokenizer instance.
        max_length: Maximum sequence length.

    Returns:
        A HuggingFace datasets.Dataset.
    """
    from datasets import Dataset as HFDataset, Features, Sequence, Value

    # Prepare flat lists for HF Dataset
    all_tokens = [record["tokens"] for record in data]
    all_labels = [
        [label2id.get(lbl, 0) for lbl in record["bio_labels"]]
        for record in data
    ]

    hf_dataset = HFDataset.from_dict({
        "tokens": all_tokens,
        "ner_tags": all_labels,
    })

    def tokenize_and_align(examples):
        """Batch tokenization + label alignment for HF Trainer."""
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        all_aligned_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = []
            prev_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    aligned.append(-100)
                elif word_id != prev_word_id:
                    aligned.append(labels[word_id])
                else:
                    # Continuation subword
                    orig_label_id = labels[word_id]
                    orig_label_str = id2label.get(orig_label_id, "O")
                    if orig_label_str.startswith("B-"):
                        entity_type = orig_label_str[2:]
                        aligned.append(label2id[f"I-{entity_type}"])
                    else:
                        aligned.append(orig_label_id)
                prev_word_id = word_id

            all_aligned_labels.append(aligned)

        tokenized["labels"] = all_aligned_labels
        return tokenized

    # Apply tokenization and alignment
    hf_dataset = hf_dataset.map(
        tokenize_and_align,
        batched=True,
        remove_columns=hf_dataset.column_names,
    )

    return hf_dataset
