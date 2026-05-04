# Arabic Sentiment Analysis — HARD Dataset

> End-to-end Arabic NLP pipeline for hotel review sentiment classification.
> Trained on balanced data, evaluated on real-world unbalanced distribution.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Experimental Design](#experimental-design)
- [Pipeline](#pipeline)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Results](#results)
- [Error Analysis](#error-analysis)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## Overview

This project builds a full Arabic NLP pipeline to classify hotel reviews as **positive** or **negative** using the **HARD** (Hotel Arabic Reviews Dataset) — ~400K Arabic reviews scraped from Booking.com.

The key design decision is **Experiment C**: rather than training and testing on the same distribution (the standard tutorial approach), we train on a class-balanced split and evaluate on the full unbalanced dataset. This simulates real-world deployment where positive reviews naturally outnumber negative ones, and directly tests whether the model generalizes beyond ideal conditions.

---

## Dataset

**HARD — Hotel Arabic Reviews Dataset**

| Property | Detail |
|---|---|
| Source | Booking.com (Arabic hotel reviews) |
| Total reviews | ~400,000 |
| Original labels | 1–5 star ratings |
| Binarization rule | 1–2 → Negative · 4–5 → Positive · 3 dropped |
| Language | Arabic (primarily Gulf dialect — UAE/Saudi) |
| Splits provided | Balanced · Unbalanced |

**Splits used in this project:**

| Split | Size | Role |
|---|---|---|
| Balanced (80%) | 84,504 | Training |
| Balanced (20%) | 21,126 | Validation |
| Unbalanced (100%) | 328,981 | Final Test — Experiment C |

---

## Experimental Design

```
Balanced Dataset (105,630 reviews)
        │
        ├── 80% ──► Train Set (84,504)  ──► Model Training
        │                                        │
        └── 20% ──► Val Set  (21,126)  ──► Hyperparameter tuning
                                                 │
                                                 ▼
Unbalanced Dataset (328,981 reviews) ──► Final Evaluation
        │                                  (Experiment C)
        └── Never seen during training
```

**Research question:**
> Does a model trained on class-balanced data generalize to a real-world skewed distribution?

This design is more rigorous than standard train/test splits because it explicitly measures the performance degradation introduced by real-world class imbalance — a critical consideration for any production NLP system.

---

## Pipeline

```
Raw HARD Data (.txt, UTF-16)
         │
         ▼
  1. Load & Parse
     (fix encoding, tab delimiters, binarize labels)
         │
         ▼
  2. EDA
     (class distribution, length analysis, Arabic wordclouds)
         │
         ▼
  3. Preprocessing
     (noise removal → normalization → diacritics removal)
         │
         ├──────────────────────────┐
         ▼                          ▼
  4A. TF-IDF + LR            4B. AraBERT Fine-tuning
  (baseline model)           (aubmindlab/bert-base-arabertv02)
         │                          │
         └──────────┬───────────────┘
                    ▼
  5. Evaluation — Experiment C
     (test on unbalanced set, compare models)
                    │
                    ▼
  6. Streamlit Demo
     (real-time Arabic sentiment prediction)
```

---

## Preprocessing

Two separate preprocessing pipelines — one per model type.

**TF-IDF Pipeline** (full normalization):

| Step | Operation |
|---|---|
| 1 | Remove URLs, HTML tags |
| 2 | Remove English characters and digits |
| 3 | Remove punctuation (Arabic + Latin) |
| 4 | Normalize alef variants: `أ إ آ` → `ا` |
| 5 | Normalize ya: `ى` → `ي` |
| 6 | Normalize ta marbuta: `ة` → `ه` |
| 7 | Remove diacritics via `pyarabic.araby.strip_tashkeel()` |

**AraBERT Pipeline** (lighter — model handles normalization internally):

| Step | Operation |
|---|---|
| 1 | Remove URLs, HTML tags |
| 2 | Remove English characters and digits |
| 3 | Remove punctuation |

---

## Models

### Baseline — TF-IDF + Logistic Regression

- `TfidfVectorizer(ngram_range=(1,2), max_features=50000, sublinear_tf=True)`
- `LogisticRegression(class_weight="balanced", max_iter=1000)`
- Fit on training set only — vectorizer never sees test data

### Main Model — AraBERT Fine-tuning

- Base model: `aubmindlab/bert-base-arabertv02`
- BERT pretrained on large Arabic corpora by AUB Mind Lab
- Fine-tuned for binary sequence classification
- Training configuration:

| Hyperparameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Epochs | 3 |
| Batch size | 16 |
| Max token length | 128 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | Linear warmup (10%) |
| Mixed precision | fp16 (autocast + GradScaler) |
| Hardware | Tesla T4 (Google Colab) |
| Training time | ~40 minutes |

---

## Results

### Validation Performance (Balanced Distribution)

| Epoch | Train Loss | Val F1 Weighted | Val F1 Negative | Val F1 Positive |
|---|---|---|---|---|
| 1 | 0.1841 | 0.9579 | 0.9569 | 0.9588 |
| 2 | 0.1221 | **0.9626** | **0.9622** | **0.9629** |
| 3 | 0.0935 | 0.9620 | 0.9616 | 0.9624 |

Best checkpoint: **Epoch 2** (F1 weighted = 0.9626)

---

### Experiment C — Test on Unbalanced Dataset (Real-World Distribution)

| Model | Accuracy | F1 Weighted | F1 Negative | F1 Positive |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | — | — | — | — |
| **AraBERT (bert-base-arabertv02)** | **0.97** | **0.9734** | **0.9196** | **0.9837** |

*TF-IDF results to be filled in after running notebook 03.*

**Per-class breakdown — AraBERT on unbalanced test set:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.88 | 0.96 | 0.92 | 52,819 |
| Positive | 0.99 | 0.97 | 0.98 | 276,162 |
| **Weighted avg** | **0.97** | **0.97** | **0.97** | **328,981** |

**Key finding:** The model trained on balanced data achieves 97% weighted F1 on the real-world unbalanced distribution, with the minority class (negative reviews) still achieving 0.92 F1 — demonstrating strong generalization despite the class shift between training and test distributions.

---

## Error Analysis

Manual inspection of misclassified examples from the unbalanced test set revealed common failure patterns:

- **Short ambiguous reviews** — very brief reviews (1–3 words) that lack sufficient context for confident classification
- **Mixed Arabic/English text** — code-switching confuses the model despite noise removal preprocessing
- **Sarcasm** — positive-sounding language used sarcastically in negative reviews is misclassified
- **Gulf dialect gap** — some dialectal expressions are not well-represented in AraBERT's MSA-heavy pretraining corpus
- **Rating-review mismatch** — labeling noise in the original dataset where the written text sentiment does not match the star rating used as label

---

## Project Structure

```
arabic-sentiment-hard/
├── data/
│   ├── balanced/
│   │   ├── balanced-reviews.txt          ← raw data (not tracked by git)
│   │   ├── preprocessed_tfidf.csv        ← generated by notebook 02
│   │   └── preprocessed_arabert.csv      ← generated by notebook 02
│   ├── unbalanced/
│   │   ├── unbalanced-reviews.txt        ← raw data (not tracked by git)
│   │   ├── preprocessed_tfidf.csv        ← generated by notebook 02
│   │   └── preprocessed_arabert.csv      ← generated by notebook 02
│   ├── cm_arabert_experimentC.png
│   ├── training_history.png
│   └── model_comparison.png
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_arabert_model.ipynb
├── src/
│   ├── preprocess.py
│   └── evaluate.py
├── app/
│   └── streamlit_app.py
├── models/                               ← not tracked by git
│   ├── arabert-sentiment/
│   ├── tfidf_vectorizer.joblib
│   ├── lr_model.joblib
│   └── metrics_arabert.json
└── README.md
```

---

## How to Run

### Install Dependencies

```bash
git clone https://github.com/adhamashraf7788/arabic-sentiment-hard
cd arabic-sentiment-hard

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers pandas numpy scikit-learn streamlit \
            joblib matplotlib seaborn tqdm pyarabic wordcloud sentencepiece
```

### Run Notebooks in Order

```bash
jupyter notebook
# Run: 01_eda → 02_preprocessing → 03_baseline_model → 04_arabert_model
# Note: notebook 04 requires a GPU — use Google Colab if needed
```

### Launch Streamlit Demo

```bash
cd app
streamlit run streamlit_app.py
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Framework | PyTorch 2.6, Hugging Face Transformers |
| NLP Model | AraBERT (aubmindlab/bert-base-arabertv02) |
| Baseline | Scikit-learn (TF-IDF + Logistic Regression) |
| Arabic NLP | pyarabic |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Demo | Streamlit |
| Training | Google Colab (Tesla T4 GPU) |
| Version Control | Git, GitHub |

---

*Built by Adham Ashraf — Arabic NLP · Deep Learning · NLP Engineering*
