# Project Report

## 1. Problem Statement

This project addresses Named Entity Recognition (NER) for medical text, specifically the extraction of drug-related entities such as drug names, dosages, frequencies, durations, routes, and forms from prescription-like sentences. In healthcare, unstructured text often hides critical medication information, and manual extraction is slow, error-prone, and difficult to scale. The objective is to build an automated system that identifies and structures these entities reliably for downstream clinical or pharmacy applications. The project compares a classical CRF baseline with a domain-adapted BioBERT transformer to improve extraction quality in a limited-data setting.

## 2. Methodology

### 2.1 Dataset

The project uses a custom medical NER dataset stored in the repository under `data/raw`, `data/train`, `data/validation`, and `data/test`. The raw annotations file contains 45 manually created medical sentences, with each sentence labeled for six entity categories: DRUG, DOSAGE, FREQUENCY, DURATION, ROUTE, and FORM. The dataset is split using a 70/15/15 ratio into train, validation, and test sets, following the configuration in `configs/config.yaml`.

### 2.2 Preprocessing

The preprocessing pipeline includes annotation conversion to BIO format, token alignment for sequence labeling, and dataset preparation for model training. Input text is normalized and split into token sequences while preserving label boundaries so that each token can be mapped to a BIO tag such as B-DRUG or I-DOSAGE. For transformer-based training, the data is processed with tokenizer-aware subword alignment using WordPiece tokenization, ensuring that each subword inherits the correct label from its source token. The project also includes domain-aware logic for medical entities such as dosage units, routes, frequencies, and forms.

### 2.3 Model / Approach

The project implements two approaches:

- CRF baseline: A Conditional Random Field model using handcrafted features such as capitalization, digit detection, dosage-unit detection, route terms, form terms, frequency terms, and surrounding context tokens.
- BioBERT transformer: A fine-tuned biomedical transformer based on `dmis-lab/biobert-base-cased-v1.2`, trained for token classification with 13 BIO labels (O + 12 entity labels). The training setup uses a learning rate of `2e-5`, batch size of 16, weight decay of 0.01, early stopping with patience 3, and a maximum of 20 epochs.

### 2.4 Tools & Frameworks

The project is built using Python and uses the following libraries and frameworks:

- PyTorch
- Hugging Face Transformers
- Transformers token classification pipeline and Trainer API
- scikit-learn and sklearn-crfsuite
- seqeval for entity-level evaluation
- FastAPI for the inference API
- JavaScript/HTML/CSS frontend
- YAML-based configuration management

## 3. Preliminary Results

The current evaluation file in `outputs/evaluation/crf/evaluation_report.json` reports the transformer-based model results for the medical NER task. The overall entity-level metrics are as follows:

| Model / Setting | Metric | Value |
| --- | --- | --- |
| BioBERT fine-tuned medical NER | F1-score | 0.957 |
| BioBERT fine-tuned medical NER | Precision | 0.943 |
| BioBERT fine-tuned medical NER | Recall | 0.971 |

These numbers indicate strong entity extraction performance on the project’s synthetic medical dataset, with balanced precision and recall. The repository also includes a CRF baseline as a comparison model, but the finalized CRF evaluation metrics were not recorded in the current evaluation artifact. The project therefore currently demonstrates the transformer model as the strongest-performing approach.

## 4. Next Steps

- Improve the dataset by adding more realistic clinical and prescription examples.
- Compare CRF and BioBERT results more systematically with a finalized baseline evaluation report.
- Perform deeper error analysis on missed or misclassified entities such as boundary mismatches and label confusion.
- Expand data augmentation and domain-specific annotation coverage for rare drug names and abbreviations.
- Finalize the project report and deployment-ready API evaluation for end-to-end usage.
