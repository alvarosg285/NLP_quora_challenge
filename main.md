# NLP Deliverable 1 — Quora Question Pairs

## 1. Objective

Build a binary classifier that determines whether two questions on Quora are
semantically equivalent (duplicates).  We present a **baseline** model and an
**improved** model, then compare them across train / validation / test splits.

Dataset: [Quora Question Pairs — Kaggle](https://www.kaggle.com/c/quora-question-pairs/overview)

---

## 2. Work distribution

| Member | Responsibility |
|--------|---------------|
| Classmate | Baseline pipeline: `cast_list_as_strings`, `get_features_from_df`, `get_mistakes`, initial `train_models.ipynb` cells |
| Student | All `# ADDED` functions in `utils.py`; extended `train_models.ipynb`; `reproduce_results.ipynb`; `utils_student.ipynb`; this document |

---

## 3. Approach

### 3.1 Baseline

| Component | Choice |
|-----------|--------|
| Tokenisation | Whitespace split, NaN → string `"nan"` |
| Representation | `CountVectorizer` (unigrams), q1 and q2 concatenated |
| Classifier | `LogisticRegression(solver="liblinear", random_state=123)` |

### 3.2 Limitations of the Baseline

- **No semantic similarity**: synonyms ("car" / "automobile") score zero overlap.
- **Word order ignored**: BoW cannot distinguish "A beats B" from "B beats A".
- **No cross-question signal**: the model never explicitly sees what q1 and q2 *share*.
- **Stopword noise**: high-frequency words ("what", "is") carry as much weight as content words.
- **High dimensionality / sparsity**: >100 k features, most zero.

### 3.3 Improved Model

We keep the BoW representation and add five handcrafted features that
directly encode the *relationship* between q1 and q2.

| # | Feature | Formula | Implementation |
|---|---------|---------|---------------|
| 0 | Jaccard similarity | $|A\cap B|/|A\cup B|$ | from scratch (Python sets) |
| 1 | Length ratio | $\min(l_1,l_2)/\max(l_1,l_2)$ | from scratch |
| 2 | Common-word F1 | $2|A\cap B|/(|A|+|B|)$ | from scratch |
| 3 | Char-bigram Dice coefficient | $2|bg_1\cap bg_2|/(|bg_1|+|bg_2|)$ | from scratch (`Counter`) |
| 4 | TF-IDF cosine similarity | $\mathbf{u}\cdot\mathbf{v}/(||\mathbf{u}||\,||\mathbf{v}||)$ | from scratch (sparse ops) |

A `TfidfVectorizer(ngram_range=(1,1), min_df=3, sublinear_tf=True)` is fitted
on training questions only (no leakage) to supply the IDF weights for feature 4.

The improved classifier is `LogisticRegression(solver="liblinear", random_state=123)`
trained on the concatenation `[X_BoW | X_handcrafted]`.

Additionally, `lcs_ratio` (word-level Longest Common Subsequence) is implemented
from scratch in `utils.py` and demonstrated in `utils_student.ipynb`; due to its
O(m·n) per-pair complexity it is not used in the full-dataset pipeline.

### 3.4 SBERT Model — Semantic Embeddings  <!-- ADDED -->

The core limitation of both models above is that they are **lexical**: two
questions sharing many tokens score high regardless of meaning, and two
semantically identical questions using different vocabulary (e.g. *"car"* vs
*"automobile"*) score low.

The SBERT model addresses this by replacing the BoW representation with
**Sentence-BERT** embeddings.

**Why SBERT and not the other candidate transformers?**

| Architecture | Inference mode | Complexity | Suitability for Quora QQ |
|---|---|---|---|
| **RoBERTa** (cross-encoder) | Joint [q1 SEP q2] pass | O(n²) per pair — too slow | ✗ without GPU |
| **DistilBERT** | Single-sentence encoder | O(n), but no paraphrase fine-tuning | ✗ weaker embeddings |
| **SBERT** `paraphrase-MiniLM-L6-v2` | Bi-encoder (Siamese) | O(n), paraphrase-trained | ✓ **chosen** |
| **BART** | Seq-to-seq generator | Not designed for embeddings | ✗ wrong task |

SBERT (Reimers & Gurevych, 2019) uses a Siamese network to fine-tune a
Transformer so that semantically similar sentences land close together in the
embedding space. The `paraphrase-MiniLM-L6-v2` checkpoint is explicitly
trained on paraphrase detection corpora — exactly the Quora task.

**Feature design** (`utils.get_sbert_interaction_features`):

Mirrors `get_interaction_features_from_df` to keep the architecture consistent:

| Column range | Feature | Intuition |
|---|---|---|
| `[0 : 384)` | `\|emb_q1 − emb_q2\|` | Captures *what differs* semantically |
| `[384 : 768)` | `emb_q1 ⊙ emb_q2` | Captures *what is shared* semantically |

A `LogisticRegression(solver="liblinear", random_state=123)` is trained on
this 768-dimensional dense matrix (same classifier family as the other models).

**Reproducibility**: encoded feature matrices are saved as `.npy` files in
`models/` by `train_models.ipynb` so `reproduce_results.ipynb` can evaluate
without re-running the encoder (the SBERT forward pass takes ~1–2 min on CPU
for 300 k questions; evaluating the LogisticRegression is instantaneous).

---

## 4. Results

*(Populated after running `reproduce_results.ipynb`)*

| model | split | roc_auc | precision | recall | f1 |
|-------|-------|---------|-----------|--------|----|
| baseline | train | — | — | — | — |
| baseline | val | — | — | — | — |
| baseline | test | — | — | — | — |
| improved | train | — | — | — | — |
| improved | val | — | — | — | — |
| improved | test | — | — | — | — |
| sbert | train | — | — | — | — |  <!-- ADDED -->
| sbert | val | — | — | — | — |    <!-- ADDED -->
| sbert | test | — | — | — | — |   <!-- ADDED -->

---

## 5. Repository structure

```
Name_Surname.zip
├── main.pdf              ← this document
├── environment.yml       ← conda environment (Python 3.9 + sentence-transformers)
├── utils.py              ← shared utility functions
├── train_models.ipynb    ← trains and saves models to models/
├── reproduce_results.ipynb ← loads models, evaluates, displays metrics
├── utils_student.ipynb   ← explanation + demos of added utils functions
└── models/               ← created by train_models.ipynb
    ├── count_vectorizer.pkl
    ├── tfidf_vectorizer.pkl
    ├── baseline_logistic.pkl
    ├── improved_logistic.pkl
    ├── sbert_logistic.pkl      ← ADDED: LogisticRegression on SBERT features
    ├── sbert_X_train.npy       ← ADDED: pre-computed SBERT features (train)
    ├── sbert_X_val.npy         ← ADDED: pre-computed SBERT features (val)
    └── sbert_X_test.npy        ← ADDED: pre-computed SBERT features (test)
```

> **Data path:** `~/Datasets/QuoraQuestionPairs/quora_data.csv`  
> The data directory is **not** included in the zip.

---

## 6. How to reproduce

```bash
# 1. Create environment
conda env create -f environment.yml --name quora_challenge_env
conda activate quora_challenge_env

# 2. Train (only needed once — safe to re-run, skips existing files)
jupyter nbconvert --to notebook --execute train_models.ipynb

# 3. Evaluate
jupyter nbconvert --to notebook --execute reproduce_results.ipynb
```