"""
test_preprocessing.py — Tests for Text Preprocessing Module
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.preprocessing.preprocessor import (
    normalize_whitespace,
    normalize_unicode,
    preprocess_text,
)


class TestWhitespaceNormalization:

    def test_multiple_spaces(self):
        assert normalize_whitespace("Take  Paracetamol") == "Take Paracetamol"

    def test_tabs_and_newlines(self):
        assert normalize_whitespace("Take\tParacetamol\n500 mg") == "Take Paracetamol 500 mg"

    def test_leading_trailing_spaces(self):
        assert normalize_whitespace("  Take Paracetamol  ") == "Take Paracetamol"

    def test_normal_text_unchanged(self):
        text = "Take Paracetamol 500 mg"
        assert normalize_whitespace(text) == text


class TestUnicodeNormalization:

    def test_em_dash(self):
        assert "\u2014" not in normalize_unicode("Take\u2014Paracetamol")

    def test_smart_quotes(self):
        result = normalize_unicode("\u201cParacetamol\u201d")
        assert "\u201c" not in result
        assert "\u201d" not in result

    def test_non_breaking_space(self):
        assert "\u00a0" not in normalize_unicode("500\u00a0mg")


class TestPreprocessText:

    def test_full_pipeline(self):
        text = "  Take\u2014Paracetamol  500\u00a0mg  "
        result = preprocess_text(text)
        assert result == "Take-Paracetamol 500 mg"

    def test_empty_string(self):
        assert preprocess_text("") == ""

    def test_none_input(self):
        assert preprocess_text(None) == ""

    def test_normal_text_preserved(self):
        text = "Take Paracetamol 500 mg twice daily for 5 days."
        assert preprocess_text(text) == text
