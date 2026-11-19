"""
error_analysis.py — NER Error Analysis Module

Identifies and categorizes prediction errors to understand model weaknesses.

Common error types in medical NER:
    1. Boundary errors — entity partially captured ("500" instead of "500 mg")
    2. Type confusion — entity detected but with wrong type (DRUG vs FORM)
    3. False negatives — entity completely missed
    4. False positives — spurious entity predicted where none exists
    5. Abbreviation errors — model fails on abbreviated forms
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_entities_from_bio(
    tokens: List[str],
    labels: List[str],
) -> List[Dict[str, Any]]:
    """
    Reconstruct entities from BIO-tagged token sequences.

    Scans the label sequence for B- tags (start of entity) and collects
    subsequent I- tags with the same entity type (continuation).

    Args:
        tokens: List of token strings.
        labels: List of BIO label strings.

    Returns:
        List of entity dicts: {"text": str, "label": str, "tokens": list}
    """
    entities = []
    current_entity = None

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            # Save any previous entity
            if current_entity:
                entities.append(current_entity)
            # Start a new entity
            entity_type = label[2:]
            current_entity = {
                "text": token,
                "label": entity_type,
                "tokens": [token],
                "start_idx": i,
                "end_idx": i,
            }
        elif label.startswith("I-") and current_entity:
            entity_type = label[2:]
            if entity_type == current_entity["label"]:
                # Continue the current entity
                current_entity["text"] += f" {token}"
                current_entity["tokens"].append(token)
                current_entity["end_idx"] = i
            else:
                # Type mismatch — save current and start new
                entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "label": entity_type,
                    "tokens": [token],
                    "start_idx": i,
                    "end_idx": i,
                }
        else:
            # O tag or I- without preceding B-
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    # Don't forget the last entity
    if current_entity:
        entities.append(current_entity)

    return entities


def analyze_errors(
    tokens_list: List[List[str]],
    true_labels_list: List[List[str]],
    pred_labels_list: List[List[str]],
    texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Perform detailed error analysis on NER predictions.

    Categorizes errors into:
    - false_positives:   predicted entity not in true entities
    - false_negatives:   true entity not predicted
    - boundary_errors:   entity detected with correct type but wrong boundary
    - type_errors:       entity boundary correct but wrong type

    Args:
        tokens_list:      List of token sequences.
        true_labels_list: List of true BIO label sequences.
        pred_labels_list: List of predicted BIO label sequences.
        texts:            Optional original text strings.

    Returns:
        Error analysis report dictionary.
    """
    all_errors = []
    error_counts = {
        "false_positives": 0,
        "false_negatives": 0,
        "boundary_errors": 0,
        "type_errors": 0,
        "correct": 0,
        "total_true": 0,
        "total_pred": 0,
    }

    for idx, (tokens, true_labels, pred_labels) in enumerate(
        zip(tokens_list, true_labels_list, pred_labels_list)
    ):
        true_entities = extract_entities_from_bio(tokens, true_labels)
        pred_entities = extract_entities_from_bio(tokens, pred_labels)

        error_counts["total_true"] += len(true_entities)
        error_counts["total_pred"] += len(pred_entities)

        original_text = texts[idx] if texts else " ".join(tokens)

        # Track which entities have been matched
        matched_true = set()
        matched_pred = set()

        # Check for exact matches and partial matches
        for ti, true_ent in enumerate(true_entities):
            for pi, pred_ent in enumerate(pred_entities):
                if pi in matched_pred:
                    continue

                # Exact match (text + label)
                if (true_ent["text"] == pred_ent["text"] and
                        true_ent["label"] == pred_ent["label"]):
                    matched_true.add(ti)
                    matched_pred.add(pi)
                    error_counts["correct"] += 1
                    break

                # Type error (same text, different label)
                if (true_ent["text"] == pred_ent["text"] and
                        true_ent["label"] != pred_ent["label"]):
                    matched_true.add(ti)
                    matched_pred.add(pi)
                    error_counts["type_errors"] += 1
                    all_errors.append({
                        "type": "type_error",
                        "sentence": original_text,
                        "true_text": true_ent["text"],
                        "true_label": true_ent["label"],
                        "pred_label": pred_ent["label"],
                        "explanation": (
                            f'"{true_ent["text"]}" labeled as '
                            f'{pred_ent["label"]} instead of {true_ent["label"]}'
                        ),
                    })
                    break

                # Boundary error (overlapping text, same label)
                if (true_ent["label"] == pred_ent["label"] and
                        _entities_overlap(true_ent, pred_ent)):
                    matched_true.add(ti)
                    matched_pred.add(pi)
                    error_counts["boundary_errors"] += 1
                    all_errors.append({
                        "type": "boundary_error",
                        "sentence": original_text,
                        "true_text": true_ent["text"],
                        "pred_text": pred_ent["text"],
                        "label": true_ent["label"],
                        "explanation": (
                            f'Entity "{true_ent["text"]}" ({true_ent["label"]}) '
                            f'predicted as "{pred_ent["text"]}"'
                        ),
                    })
                    break

        # Remaining unmatched true entities → false negatives
        for ti, true_ent in enumerate(true_entities):
            if ti not in matched_true:
                error_counts["false_negatives"] += 1
                all_errors.append({
                    "type": "false_negative",
                    "sentence": original_text,
                    "missed_text": true_ent["text"],
                    "missed_label": true_ent["label"],
                    "explanation": (
                        f'Missed entity "{true_ent["text"]}" ({true_ent["label"]})'
                    ),
                })

        # Remaining unmatched predicted entities → false positives
        for pi, pred_ent in enumerate(pred_entities):
            if pi not in matched_pred:
                error_counts["false_positives"] += 1
                all_errors.append({
                    "type": "false_positive",
                    "sentence": original_text,
                    "spurious_text": pred_ent["text"],
                    "spurious_label": pred_ent["label"],
                    "explanation": (
                        f'Spurious entity "{pred_ent["text"]}" '
                        f'predicted as {pred_ent["label"]}'
                    ),
                })

    report = {
        "summary": error_counts,
        "errors": all_errors,
    }

    return report


def _entities_overlap(ent1: Dict, ent2: Dict) -> bool:
    """Check if two entities have overlapping token indices."""
    range1 = set(range(ent1["start_idx"], ent1["end_idx"] + 1))
    range2 = set(range(ent2["start_idx"], ent2["end_idx"] + 1))
    return bool(range1 & range2)


def format_error_report(report: Dict[str, Any]) -> str:
    """
    Format the error analysis report as a human-readable string.

    Args:
        report: Error analysis report from analyze_errors().

    Returns:
        Formatted report string.
    """
    lines = []
    summary = report["summary"]

    lines.append("=" * 60)
    lines.append("ERROR ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total true entities:     {summary['total_true']}")
    lines.append(f"Total predicted entities: {summary['total_pred']}")
    lines.append(f"Correct predictions:     {summary['correct']}")
    lines.append(f"False positives:         {summary['false_positives']}")
    lines.append(f"False negatives:         {summary['false_negatives']}")
    lines.append(f"Boundary errors:         {summary['boundary_errors']}")
    lines.append(f"Type errors:             {summary['type_errors']}")
    lines.append("")

    if report["errors"]:
        lines.append("-" * 60)
        lines.append("DETAILED ERRORS")
        lines.append("-" * 60)

        for i, error in enumerate(report["errors"][:20], 1):  # Limit to 20
            lines.append(f"\nError #{i} ({error['type']}):")
            lines.append(f"  Sentence: {error.get('sentence', 'N/A')}")
            lines.append(f"  Details:  {error['explanation']}")

    return "\n".join(lines)


def save_error_report(
    report: Dict[str, Any],
    output_dir: str,
) -> None:
    """Save error analysis report to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    with open(output_path / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save human-readable report
    with open(output_path / "error_analysis.txt", "w", encoding="utf-8") as f:
        f.write(format_error_report(report))

    logger.info(f"Error analysis saved to: {output_path}")
