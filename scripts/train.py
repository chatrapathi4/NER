"""
train.py — Unified Training Script

Trains either the CRF baseline or BioBERT transformer model.

Usage:
    python scripts/train.py --model crf
    python scripts/train.py --model transformer
    python scripts/train.py --model transformer --epochs 10 --batch-size 8
"""

import sys
import json
import logging
import argparse
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Medical NER model")
    parser.add_argument(
        "--model",
        choices=["crf", "transformer"],
        required=True,
        help="Model type to train",
    )
    parser.add_argument(
        "--train-file",
        default="data/train/train.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--val-file",
        default="data/validation/val.jsonl",
        help="Path to validation data",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_path = str(PROJECT_ROOT / args.train_file)
    val_path = str(PROJECT_ROOT / args.val_file)

    # Check that data files exist
    if not Path(train_path).exists():
        logger.error(
            f"Training data not found: {train_path}\n"
            "Run 'python scripts/prepare_data.py' first."
        )
        sys.exit(1)

    if args.model == "crf":
        from src.training.train_crf import train_crf

        results = train_crf(
            train_path=train_path,
            val_path=val_path if Path(val_path).exists() else None,
            model_save_path=str(PROJECT_ROOT / "models" / "crf" / "crf_model.pkl"),
        )

    elif args.model == "transformer":
        from src.training.train_transformer import train_transformer

        results = train_transformer(
            train_path=train_path,
            val_path=val_path if Path(val_path).exists() else None,
            output_dir=str(PROJECT_ROOT / "models" / "biobert"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
        )

    # Save results
    results_dir = PROJECT_ROOT / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"train_results_{args.model}.json"

    with open(results_file, "w") as f:
        # Convert non-serializable values
        serializable = {}
        for k, v in results.items():
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = str(v)
        json.dump(serializable, f, indent=2)

    logger.info(f"\nTraining results saved to: {results_file}")
    logger.info("✓ Training complete!")


if __name__ == "__main__":
    main()
