"""
prepare_data.py — Data Preparation Pipeline

Converts raw span-annotated data into BIO-tagged format and splits it
into train/validation/test sets.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --augment     # With data augmentation
"""

import sys
import json
import random
import logging
import argparse
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.annotation import (
    load_annotations,
    convert_annotations_to_bio,
    save_bio_data,
    validate_bio_sequence,
)
from src.data.augmentation import augment_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare Medical NER data")
    parser.add_argument(
        "--raw-file",
        default="data/raw/annotations.jsonl",
        help="Path to raw annotations JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Base output directory",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Apply data augmentation to training set",
    )
    parser.add_argument(
        "--num-augments", type=int, default=2,
        help="Number of augmented copies per sentence",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Medical NER — Data Preparation Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Load raw annotations ──
    raw_path = PROJECT_ROOT / args.raw_file
    logger.info(f"Loading raw annotations from: {raw_path}")
    annotations = load_annotations(str(raw_path))
    logger.info(f"Loaded {len(annotations)} annotated sentences")

    # ── Step 2: Convert to BIO format ──
    logger.info("Converting span annotations to BIO format...")
    bio_data = convert_annotations_to_bio(annotations)

    # Validate BIO sequences
    for record in bio_data:
        record["bio_labels"] = validate_bio_sequence(
            record["tokens"], record["bio_labels"]
        )

    # Save processed BIO data
    processed_path = PROJECT_ROOT / args.output_dir / "processed" / "bio_dataset.jsonl"
    save_bio_data(bio_data, str(processed_path))
    logger.info(f"Saved {len(bio_data)} BIO-tagged sentences to: {processed_path}")

    # ── Step 3: Log entity statistics ──
    entity_counts = {}
    for record in bio_data:
        for label in record["bio_labels"]:
            if label != "O":
                prefix = label.split("-")[1] if "-" in label else label
                entity_counts[prefix] = entity_counts.get(prefix, 0) + 1

    logger.info("\nEntity token counts:")
    for entity, count in sorted(entity_counts.items()):
        logger.info(f"  {entity:<15} {count}")

    # ── Step 4: Split into train/val/test ──
    random.seed(args.seed)
    indices = list(range(len(bio_data)))
    random.shuffle(indices)

    n = len(indices)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_data = [bio_data[i] for i in train_indices]
    val_data = [bio_data[i] for i in val_indices]
    test_data = [bio_data[i] for i in test_indices]

    logger.info(f"\nSplit sizes (seed={args.seed}):")
    logger.info(f"  Train:      {len(train_data)} sentences")
    logger.info(f"  Validation: {len(val_data)} sentences")
    logger.info(f"  Test:       {len(test_data)} sentences")

    # ── Step 5: Data augmentation (training set only) ──
    if args.augment:
        original_count = len(train_data)
        logger.info(f"\nApplying data augmentation (num_augments={args.num_augments})...")
        train_data = augment_dataset(
            train_data,
            num_augments=args.num_augments,
            seed=args.seed,
        )
        logger.info(
            f"Training data: {original_count} → {len(train_data)} "
            f"(+{len(train_data) - original_count} augmented)"
        )

    # ── Step 6: Save splits ──
    output_base = PROJECT_ROOT / args.output_dir

    train_path = output_base / "train" / "train.jsonl"
    val_path = output_base / "validation" / "val.jsonl"
    test_path = output_base / "test" / "test.jsonl"

    save_bio_data(train_data, str(train_path))
    save_bio_data(val_data, str(val_path))
    save_bio_data(test_data, str(test_path))

    logger.info(f"\nSaved splits:")
    logger.info(f"  Train:      {train_path}")
    logger.info(f"  Validation: {val_path}")
    logger.info(f"  Test:       {test_path}")

    logger.info("\n✓ Data preparation complete!")

    # ── Print sample ──
    logger.info("\nSample BIO-tagged sentence:")
    sample = train_data[0]
    for token, label in zip(sample["tokens"], sample["bio_labels"]):
        logger.info(f"  {token:<20} {label}")


if __name__ == "__main__":
    main()
