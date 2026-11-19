"""
train_crf.py — CRF Model Training Script

Trains the CRF baseline model using hand-crafted features.
CRF training is fast (seconds) and requires no GPU.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.data.annotation import load_bio_data
from src.models.crf_model import (
    CRFModel,
    extract_features_from_data,
)

logger = logging.getLogger(__name__)


def train_crf(
    train_path: str,
    val_path: Optional[str] = None,
    model_save_path: str = "models/crf/crf_model.pkl",
    algorithm: str = "lbfgs",
    c1: float = 0.1,
    c2: float = 0.1,
    max_iterations: int = 100,
) -> dict:
    """
    Train a CRF NER model.

    Steps:
        1. Load BIO-tagged training data
        2. Extract hand-crafted features for each token
        3. Train the CRF model
        4. Optionally evaluate on validation data
        5. Save the trained model

    Args:
        train_path:      Path to training data JSONL.
        val_path:        Path to validation data JSONL (optional).
        model_save_path: Where to save the trained model.
        algorithm:       CRF optimization algorithm.
        c1:              L1 regularization.
        c2:              L2 regularization.
        max_iterations:  Max training iterations.

    Returns:
        Dictionary with training results.
    """
    logger.info("=" * 60)
    logger.info("Training CRF Baseline Model")
    logger.info("=" * 60)

    # ── Step 1: Load training data ──
    logger.info(f"Loading training data from: {train_path}")
    train_data = load_bio_data(train_path)
    logger.info(f"Training samples: {len(train_data)}")

    # ── Step 2: Extract features ──
    logger.info("Extracting features...")
    X_train, y_train = extract_features_from_data(train_data)

    # ── Step 3: Train the CRF ──
    logger.info(
        f"Training CRF (algorithm={algorithm}, c1={c1}, c2={c2}, "
        f"max_iter={max_iterations})"
    )
    model = CRFModel(
        algorithm=algorithm,
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
    )
    model.train(X_train, y_train)
    logger.info("CRF training complete.")

    # ── Step 4: Evaluate on validation data ──
    results = {"model": "CRF", "train_samples": len(train_data)}

    if val_path:
        logger.info(f"Evaluating on validation data: {val_path}")
        val_data = load_bio_data(val_path)
        X_val, y_val = extract_features_from_data(val_data)
        y_pred = model.predict(X_val)

        # Import evaluation metrics
        try:
            from seqeval.metrics import (
                classification_report,
                f1_score,
                precision_score,
                recall_score,
            )

            report = classification_report(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)

            logger.info(f"\nValidation Results:\n{report}")
            logger.info(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

            results.update({
                "val_f1": f1,
                "val_precision": precision,
                "val_recall": recall,
                "val_report": report,
                "val_samples": len(val_data),
            })
        except ImportError:
            logger.warning("seqeval not installed. Skipping evaluation metrics.")

    # ── Step 5: Save the model ──
    logger.info(f"Saving CRF model to: {model_save_path}")
    model.save(model_save_path)
    logger.info("CRF model saved successfully.")

    return results
