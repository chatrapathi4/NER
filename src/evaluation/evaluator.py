"""
evaluator.py — NER Evaluation Module

Computes entity-level precision, recall, and F1-score using the seqeval
library, which is the standard evaluation framework for NER tasks.

Why entity-level evaluation (not token-level accuracy)?
    Token-level accuracy is misleading for NER because most tokens are O
    (outside any entity). If 90% of tokens are O, a model that always
    predicts O achieves 90% accuracy while extracting zero entities.

    Entity-level evaluation counts an entity as correct only if:
    1. The entity TYPE matches (e.g., DRUG, DOSAGE)
    2. The entity BOUNDARY matches exactly (all tokens from B- through I-)

Metrics explained:
    Precision = TP / (TP + FP) — of all entities predicted, how many are correct?
    Recall    = TP / (TP + FN) — of all true entities, how many were found?
    F1-score  = 2 * (P * R) / (P + R) — harmonic mean of precision and recall
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.data.annotation import id2label

logger = logging.getLogger(__name__)


def evaluate_predictions(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute entity-level NER evaluation metrics.

    Args:
        true_labels: List of true BIO label sequences (one per sentence).
        pred_labels: List of predicted BIO label sequences (one per sentence).
        output_dir:  Directory to save evaluation report (optional).

    Returns:
        Dictionary with overall and per-entity metrics.
    """
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
    from seqeval.scheme import IOB2

    # Overall metrics
    overall_f1 = f1_score(true_labels, pred_labels)
    overall_precision = precision_score(true_labels, pred_labels)
    overall_recall = recall_score(true_labels, pred_labels)

    # Detailed per-entity report
    report_str = classification_report(true_labels, pred_labels, digits=4)

    # Parse per-entity metrics from the detailed report
    report_dict = classification_report(
        true_labels, pred_labels, output_dict=True
    )

    results = {
        "overall": {
            "f1": float(overall_f1),
            "precision": float(overall_precision),
            "recall": float(overall_recall),
        },
        "per_entity": {},
        "report_text": report_str,
    }

    # Extract per-entity-type metrics
    for entity_type in ["DRUG", "DOSAGE", "FREQUENCY", "DURATION", "ROUTE", "FORM"]:
        if entity_type in report_dict:
            results["per_entity"][entity_type] = {
                "precision": float(report_dict[entity_type]["precision"]),
                "recall": float(report_dict[entity_type]["recall"]),
                "f1": float(report_dict[entity_type]["f1-score"]),
                "support": int(report_dict[entity_type]["support"]),
            }

    # Save report if output_dir provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / "evaluation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation report saved to: {report_file}")

        report_txt = output_path / "evaluation_report.txt"
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("Medical NER Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Overall F1:        {overall_f1:.4f}\n")
            f.write(f"Overall Precision: {overall_precision:.4f}\n")
            f.write(f"Overall Recall:    {overall_recall:.4f}\n\n")
            f.write("Detailed Per-Entity Report:\n")
            f.write("-" * 60 + "\n")
            f.write(report_str + "\n")
        logger.info(f"Text report saved to: {report_txt}")

    return results


def evaluate_crf(
    test_path: str,
    model_path: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained CRF model on test data.

    Args:
        test_path:  Path to test data JSONL.
        model_path: Path to saved CRF model.
        output_dir: Directory to save evaluation report.

    Returns:
        Evaluation results dictionary.
    """
    from src.data.annotation import load_bio_data
    from src.models.crf_model import CRFModel, extract_features_from_data

    logger.info("Evaluating CRF model...")
    test_data = load_bio_data(test_path)
    model = CRFModel.load(model_path)

    X_test, y_test = extract_features_from_data(test_data)
    y_pred = model.predict(X_test)

    if output_dir:
        output_dir = str(Path(output_dir) / "crf")

    results = evaluate_predictions(y_test, y_pred, output_dir)
    results["model"] = "CRF"
    results["test_samples"] = len(test_data)

    logger.info(f"\nCRF Results:\n{results['report_text']}")
    return results


def evaluate_transformer(
    test_path: str,
    model_dir: str,
    max_length: int = 128,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained transformer model on test data.

    Args:
        test_path:  Path to test data JSONL.
        model_dir:  Path to the saved transformer model directory.
        max_length: Maximum sequence length.
        output_dir: Directory to save evaluation report.

    Returns:
        Evaluation results dictionary.
    """
    import torch
    import numpy as np
    from src.data.annotation import load_bio_data
    from src.data.dataset import create_hf_dataset
    from src.models.transformer_model import load_trained_model
    from transformers import Trainer, TrainingArguments
    from src.training.train_transformer import compute_metrics

    logger.info("Evaluating Transformer model...")
    test_data = load_bio_data(test_path)

    model, tokenizer = load_trained_model(model_dir)
    test_dataset = create_hf_dataset(test_data, tokenizer, max_length)

    # Use Trainer for prediction (handles batching and device placement)
    eval_args = TrainingArguments(
        output_dir="outputs/eval_tmp",
        per_device_eval_batch_size=16,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics,
    )

    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_ids = np.argmax(predictions.predictions, axis=2)
    label_ids = predictions.label_ids

    # Convert to label strings
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(pred_ids, label_ids):
        true_sent = []
        pred_sent = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            true_sent.append(id2label.get(int(label_id), "O"))
            pred_sent.append(id2label.get(int(pred_id), "O"))
        true_labels.append(true_sent)
        pred_labels.append(pred_sent)

    if output_dir:
        output_dir = str(Path(output_dir) / "transformer")

    results = evaluate_predictions(true_labels, pred_labels, output_dir)
    results["model"] = "BioBERT"
    results["test_samples"] = len(test_data)

    logger.info(f"\nTransformer Results:\n{results['report_text']}")
    return results


def compare_models(
    crf_results: Dict[str, Any],
    transformer_results: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> str:
    """
    Generate a comparison table between CRF and Transformer models.

    Args:
        crf_results:         CRF evaluation results.
        transformer_results: Transformer evaluation results.
        output_dir:          Where to save the comparison.

    Returns:
        Formatted comparison string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MODEL COMPARISON: CRF vs BioBERT Transformer")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Metric':<20} {'CRF':<15} {'BioBERT':<15} {'Difference':<15}")
    lines.append("-" * 70)

    for metric in ["f1", "precision", "recall"]:
        crf_val = crf_results.get("overall", {}).get(metric, 0)
        tf_val = transformer_results.get("overall", {}).get(metric, 0)
        diff = tf_val - crf_val
        sign = "+" if diff > 0 else ""
        lines.append(
            f"{metric.capitalize():<20} {crf_val:<15.4f} {tf_val:<15.4f} {sign}{diff:<15.4f}"
        )

    lines.append("")
    lines.append("Per-Entity F1 Comparison:")
    lines.append("-" * 70)
    lines.append(f"{'Entity':<20} {'CRF':<15} {'BioBERT':<15}")
    lines.append("-" * 70)

    for entity in ["DRUG", "DOSAGE", "FREQUENCY", "DURATION", "ROUTE", "FORM"]:
        crf_f1 = crf_results.get("per_entity", {}).get(entity, {}).get("f1", 0)
        tf_f1 = transformer_results.get("per_entity", {}).get(entity, {}).get("f1", 0)
        lines.append(f"{entity:<20} {crf_f1:<15.4f} {tf_f1:<15.4f}")

    comparison = "\n".join(lines)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "model_comparison.txt", "w") as f:
            f.write(comparison)

    return comparison
