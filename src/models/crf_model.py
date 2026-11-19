"""
crf_model.py — CRF Baseline Model for Medical NER

Conditional Random Fields (CRF) is a classical sequence labeling model that
serves as our baseline. Unlike neural models, CRFs rely on hand-crafted
features but can still perform reasonably well for NER tasks.

Why a CRF baseline?
    1. Establishes a performance floor — if the transformer doesn't beat CRF,
       something is wrong with our transformer approach.
    2. Demonstrates that domain knowledge (via features) matters.
    3. CRFs are interpretable — we can see which features drive predictions.
    4. CRFs train in seconds, making them ideal for rapid iteration.
    5. Required in academic projects to show comparative analysis.

Feature design:
    We extract features that encode:
    - Word identity (the word itself, lowercased)
    - Word shape (capitalization, digits, punctuation patterns)
    - Prefix/suffix (first/last 2-3 characters)
    - Context (features of neighboring words)
    - Whether the word looks like a number, drug name, dosage, etc.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import joblib

try:
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics as crf_metrics
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────

def word_to_features(sent: List[str], i: int) -> Dict[str, Any]:
    """
    Extract features for a single token at position i in the sentence.

    Features capture:
        - The word itself (identity)
        - Its shape (uppercase, digit, punctuation patterns)
        - Its prefix and suffix (morphological hints)
        - Whether it looks like a number or unit
        - Context from neighboring words (±2 window)

    Args:
        sent: List of token strings.
        i:    Index of the current token.

    Returns:
        Dictionary of feature_name → feature_value.
    """
    word = sent[i]

    # ── Base features for current word ──
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:] if len(word) >= 3 else word,
        "word[-2:]": word[-2:] if len(word) >= 2 else word,
        "word[:3]": word[:3] if len(word) >= 3 else word,
        "word[:2]": word[:2] if len(word) >= 2 else word,
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.isalpha()": word.isalpha(),
        "word.length": len(word),
        # Does it look like a number (including decimals)?
        "word.is_numeric": bool(re.match(r"^\d+\.?\d*$", word)),
        # Is it a common dosage unit?
        "word.is_unit": word.lower() in {
            "mg", "mcg", "g", "ml", "units", "iu", "meq",
            "puffs", "drops", "drop", "puff",
        },
        # Is it a common route word?
        "word.is_route": word.lower() in {
            "orally", "iv", "im", "sc", "sq", "topically",
            "subcutaneously", "intravenously", "intramuscularly",
            "rectally", "sublingually",
        },
        # Is it a common form word?
        "word.is_form": word.lower() in {
            "tablet", "tablets", "capsule", "capsules", "cream",
            "gel", "ointment", "syrup", "solution", "injection",
            "inhaler", "drops", "suspension", "powder",
        },
        # Is it a common frequency word?
        "word.is_freq": word.lower() in {
            "once", "twice", "daily", "hourly", "weekly",
            "monthly", "bedtime", "morning", "evening",
            "bid", "tid", "qid", "qd", "prn",
        },
    }

    # ── Previous word features (context) ──
    if i > 0:
        prev_word = sent[i - 1]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:word.istitle()": prev_word.istitle(),
            "-1:word.isdigit()": prev_word.isdigit(),
            "-1:word.is_numeric": bool(re.match(r"^\d+\.?\d*$", prev_word)),
        })
    else:
        features["BOS"] = True  # Beginning of sentence

    # ── Next word features (context) ──
    if i < len(sent) - 1:
        next_word = sent[i + 1]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:word.istitle()": next_word.istitle(),
            "+1:word.isdigit()": next_word.isdigit(),
            "+1:word.is_unit": next_word.lower() in {
                "mg", "mcg", "g", "ml", "units",
            },
        })
    else:
        features["EOS"] = True  # End of sentence

    # ── Two-word lookahead/lookbehind ──
    if i > 1:
        features["-2:word.lower()"] = sent[i - 2].lower()
    if i < len(sent) - 2:
        features["+2:word.lower()"] = sent[i + 2].lower()

    return features


def sent_to_features(sent: List[str]) -> List[Dict[str, Any]]:
    """Extract features for every token in a sentence."""
    return [word_to_features(sent, i) for i in range(len(sent))]


def sent_to_labels(labels: List[str]) -> List[str]:
    """Extract labels for a sentence (identity function for clarity)."""
    return list(labels)


def extract_features_from_data(
    data: List[Dict[str, Any]]
) -> Tuple[List[List[Dict]], List[List[str]]]:
    """
    Extract CRF features and labels from BIO-tagged data.

    Args:
        data: List of dicts with 'tokens' and 'bio_labels'.

    Returns:
        Tuple of (X_features, y_labels).
    """
    X = [sent_to_features(record["tokens"]) for record in data]
    y = [sent_to_labels(record["bio_labels"]) for record in data]
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# CRF Model Class
# ──────────────────────────────────────────────────────────────────────────────

class CRFModel:
    """
    CRF-based NER model using sklearn-crfsuite.

    Args:
        algorithm:       Optimization algorithm (default: lbfgs).
        c1:              L1 regularization coefficient.
        c2:              L2 regularization coefficient.
        max_iterations:  Maximum training iterations.
    """

    def __init__(
        self,
        algorithm: str = "lbfgs",
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
    ):
        if not CRF_AVAILABLE:
            raise ImportError(
                "sklearn-crfsuite is required. "
                "Install it with: pip install sklearn-crfsuite"
            )

        self.crf = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )

    def train(
        self,
        X_train: List[List[Dict]],
        y_train: List[List[str]],
    ) -> None:
        """
        Train the CRF model.

        Args:
            X_train: List of feature-dict sequences.
            y_train: List of label sequences.
        """
        self.crf.fit(X_train, y_train)

    def predict(self, X: List[List[Dict]]) -> List[List[str]]:
        """
        Predict BIO labels for feature sequences.

        Args:
            X: List of feature-dict sequences.

        Returns:
            List of predicted label sequences.
        """
        return self.crf.predict(X)

    def predict_single(self, tokens: List[str]) -> List[str]:
        """
        Predict BIO labels for a single tokenized sentence.

        Args:
            tokens: List of token strings.

        Returns:
            List of predicted BIO labels.
        """
        features = sent_to_features(tokens)
        predictions = self.crf.predict([features])
        return predictions[0]

    def save(self, filepath: str) -> None:
        """Save the trained CRF model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.crf, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CRFModel":
        """Load a trained CRF model from disk."""
        model = cls.__new__(cls)
        model.crf = joblib.load(filepath)
        return model
