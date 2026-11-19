"""
train_transformer.py — BioBERT Fine-Tuning for Medical NER

Fine-tunes a pretrained BioBERT model on our medical NER dataset using
the HuggingFace Trainer API.

Training details:
    - Optimizer: AdamW (Adam with decoupled weight decay)
    - Scheduler: Linear warmup → linear decay
    - Early stopping: Monitors validation F1, stops after N epochs of no improvement
    - Checkpointing: Saves best model based on validation F1
    - Seed: Fixed for reproducibility

Why these hyperparameters?
    - Learning rate 2e-5: Standard for BERT fine-tuning (higher → unstable,
      lower → slow convergence)
    - Batch size 16: Balance between GPU memory and gradient quality
    - Weight decay 0.01: Mild L2 regularization to prevent overfitting
    - Warmup ratio 0.1: Gradually ramp up LR to avoid destroying pretrained weights
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from src.data.annotation import load_bio_data, id2label, label2id
from src.data.dataset import create_hf_dataset
from src.models.transformer_model import (
    load_tokenizer,
    load_model,
    DEFAULT_MODEL_NAME,
)

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """
    Compute entity-level precision, recall, and F1 during training.

    This function is called by the HuggingFace Trainer at each evaluation step.
    It uses seqeval to compute entity-level metrics (not token-level), which
    is the standard for NER evaluation.

    Entity-level means: an entity is only counted as correct if both the
    entity type AND the exact boundary (all tokens) match the gold label.

    Args:
        eval_pred: Tuple of (predictions, labels) from the Trainer.

    Returns:
        Dictionary of metric_name → metric_value.
    """
    from seqeval.metrics import (
        f1_score,
        precision_score,
        recall_score,
        classification_report,
    )

    predictions, labels = eval_pred

    # predictions shape: (batch, seq_len, num_labels) → argmax → (batch, seq_len)
    predictions = np.argmax(predictions, axis=2)

    # Convert integer IDs back to label strings, ignoring -100 padding
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_sent = []
        pred_sent = []

        for pred_id, label_id in zip(pred_seq, label_seq):
            # Skip tokens with label -100 (special tokens / padding)
            if label_id == -100:
                continue
            true_sent.append(id2label.get(int(label_id), "O"))
            pred_sent.append(id2label.get(int(pred_id), "O"))

        true_labels.append(true_sent)
        pred_labels.append(pred_sent)

    # Compute entity-level metrics
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def train_transformer(
    train_path: str,
    val_path: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = "models/biobert",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 3,
    max_length: int = 128,
    fp16: bool = False,
    seed: int = 42,
    gradient_accumulation_steps: int = 1,
    logging_steps: int = 10,
) -> Dict[str, Any]:
    """
    Fine-tune a BioBERT model for medical NER.

    Steps:
        1. Load the pretrained tokenizer and model
        2. Load and prepare the training/validation datasets
        3. Configure the HuggingFace Trainer with training arguments
        4. Train with early stopping
        5. Save the best model

    Args:
        train_path:     Path to training data JSONL.
        val_path:       Path to validation data JSONL.
        model_name:     HuggingFace model name/path.
        output_dir:     Where to save the trained model.
        epochs:         Maximum number of training epochs.
        batch_size:     Training batch size.
        learning_rate:  Peak learning rate.
        weight_decay:   L2 regularization coefficient.
        warmup_ratio:   Fraction of steps for LR warmup.
        early_stopping_patience: Epochs to wait for improvement.
        max_length:     Maximum sequence length.
        fp16:           Use mixed precision (requires CUDA GPU).
        seed:           Random seed.
        gradient_accumulation_steps: Accumulate gradients over N steps.
        logging_steps:  Log metrics every N steps.

    Returns:
        Dictionary with training results.
    """
    from transformers import (
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    import torch

    logger.info("=" * 60)
    logger.info("Fine-Tuning BioBERT Transformer Model")
    logger.info("=" * 60)

    # ── Step 1: Load tokenizer and model ──
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)

    logger.info(f"Loading model: {model_name}")
    model = load_model(model_name)

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cpu":
        logger.warning(
            "Training on CPU — this will be slow. "
            "Consider using a GPU or reducing epochs/data size."
        )
        # Disable fp16 on CPU
        fp16 = False

    # ── Step 2: Load and prepare datasets ──
    logger.info(f"Loading training data: {train_path}")
    train_data = load_bio_data(train_path)
    logger.info(f"Training samples: {len(train_data)}")

    train_dataset = create_hf_dataset(train_data, tokenizer, max_length)

    eval_dataset = None
    if val_path:
        logger.info(f"Loading validation data: {val_path}")
        val_data = load_bio_data(val_path)
        logger.info(f"Validation samples: {len(val_data)}")
        eval_dataset = create_hf_dataset(val_data, tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        # Evaluation strategy
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        # Best model selection
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="f1" if eval_dataset else None,
        greater_is_better=True if eval_dataset else None,
        # Logging
        logging_steps=logging_steps,
        # Optimization
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Reproducibility
        seed=seed,
        data_seed=seed,
        # Save limits
        save_total_limit=3,
        # Disable unnecessary features for simplicity
        report_to="none",
    )

    # ── Step 4: Create Trainer with callbacks ──
    callbacks = []
    if eval_dataset and early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ── Step 5: Train ──
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training complete!")

    # ── Step 6: Save the best model and tokenizer ──
    final_model_dir = Path(output_dir) / "final"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving final model to: {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    # ── Collect results ──
    results = {
        "model": "BioBERT",
        "model_name": model_name,
        "train_samples": len(train_data),
        "train_loss": train_result.training_loss,
        "epochs_trained": train_result.metrics.get(
            "epoch", epochs
        ),
    }

    # Evaluate on validation set if available
    if eval_dataset:
        eval_results = trainer.evaluate()
        results.update({
            "val_f1": eval_results.get("eval_f1", None),
            "val_precision": eval_results.get("eval_precision", None),
            "val_recall": eval_results.get("eval_recall", None),
            "val_loss": eval_results.get("eval_loss", None),
        })
        logger.info(
            f"Validation — F1: {results.get('val_f1', 'N/A'):.4f} | "
            f"Precision: {results.get('val_precision', 'N/A'):.4f} | "
            f"Recall: {results.get('val_recall', 'N/A'):.4f}"
        )

    return results
