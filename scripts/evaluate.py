"""
evaluate.py — Model Evaluation Script

Evaluates trained models on the test set and generates comparison reports.

Usage:
    python scripts/evaluate.py --model crf
    python scripts/evaluate.py --model transformer
    python scripts/evaluate.py --model both
"""

import sys
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
    parser = argparse.ArgumentParser(description="Evaluate Medical NER model")
    parser.add_argument(
        "--model",
        choices=["crf", "transformer", "both"],
        default="both",
        help="Which model(s) to evaluate",
    )
    parser.add_argument(
        "--test-file",
        default="data/test/test.jsonl",
        help="Path to test data",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory to save evaluation reports",
    )

    args = parser.parse_args()
    test_path = str(PROJECT_ROOT / args.test_file)
    output_dir = str(PROJECT_ROOT / args.output_dir)

    if not Path(test_path).exists():
        logger.error(
            f"Test data not found: {test_path}\n"
            "Run 'python scripts/prepare_data.py' first."
        )
        sys.exit(1)

    crf_results = None
    transformer_results = None

    # Evaluate CRF
    if args.model in ("crf", "both"):
        crf_model_path = str(PROJECT_ROOT / "models" / "crf" / "crf_model.pkl")
        if Path(crf_model_path).exists():
            from src.evaluation.evaluator import evaluate_crf
            crf_results = evaluate_crf(test_path, crf_model_path, output_dir)
        else:
            logger.warning(
                f"CRF model not found at {crf_model_path}. "
                "Train it first: python scripts/train.py --model crf"
            )

    # Evaluate Transformer
    if args.model in ("transformer", "both"):
        transformer_dir = str(PROJECT_ROOT / "models" / "biobert" / "final")
        if Path(transformer_dir).exists():
            from src.evaluation.evaluator import evaluate_transformer
            transformer_results = evaluate_transformer(
                test_path, transformer_dir, output_dir=output_dir,
            )
        else:
            logger.warning(
                f"Transformer model not found at {transformer_dir}. "
                "Train it first: python scripts/train.py --model transformer"
            )

    # Compare models if both are available
    if crf_results and transformer_results:
        from src.evaluation.evaluator import compare_models
        comparison = compare_models(crf_results, transformer_results, output_dir)
        logger.info(f"\n{comparison}")

    # Run error analysis if transformer results available
    if transformer_results:
        logger.info("\nRunning error analysis...")
        try:
            from src.data.annotation import load_bio_data
            from src.evaluation.error_analysis import (
                analyze_errors,
                format_error_report,
                save_error_report,
            )
            from src.models.transformer_model import load_trained_model
            from src.inference.predictor import reconstruct_entities
            import torch
            import numpy as np

            test_data = load_bio_data(test_path)
            model, tokenizer = load_trained_model(
                str(PROJECT_ROOT / "models" / "biobert" / "final")
            )

            from src.data.annotation import id2label
            from src.data.dataset import create_hf_dataset
            from transformers import Trainer, TrainingArguments

            test_dataset = create_hf_dataset(test_data, tokenizer, 128)
            eval_args = TrainingArguments(
                output_dir="outputs/eval_tmp",
                per_device_eval_batch_size=16,
                report_to="none",
                use_cpu=not torch.cuda.is_available(),
            )
            trainer = Trainer(model=model, args=eval_args)
            predictions = trainer.predict(test_dataset)
            pred_ids = np.argmax(predictions.predictions, axis=2)
            label_ids = predictions.label_ids

            tokens_list = [d["tokens"] for d in test_data]
            true_labels_list = [d["bio_labels"] for d in test_data]
            texts = [d.get("original_text", " ".join(d["tokens"])) for d in test_data]

            # Reconstruct predicted labels per sentence
            pred_labels_list = []
            for pred_seq, label_seq in zip(pred_ids, label_ids):
                pred_sent = []
                for pred_id, label_id in zip(pred_seq, label_seq):
                    if label_id == -100:
                        continue
                    pred_sent.append(id2label.get(int(pred_id), "O"))
                pred_labels_list.append(pred_sent)

            # Align lengths (truncation may cause mismatch)
            aligned_tokens = []
            aligned_true = []
            aligned_pred = []
            for tokens, true_lbls, pred_lbls in zip(
                tokens_list, true_labels_list, pred_labels_list
            ):
                min_len = min(len(tokens), len(true_lbls), len(pred_lbls))
                aligned_tokens.append(tokens[:min_len])
                aligned_true.append(true_lbls[:min_len])
                aligned_pred.append(pred_lbls[:min_len])

            report = analyze_errors(aligned_tokens, aligned_true, aligned_pred, texts)
            save_error_report(report, str(Path(output_dir) / "transformer"))
            logger.info(format_error_report(report))

        except Exception as e:
            logger.warning(f"Error analysis failed: {e}")

    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
