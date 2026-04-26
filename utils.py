import numpy as np
import scipy

def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings

def get_features_from_df(df, count_vectorizer):
    """
    returns a sparse matrix containing the features built by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    q1_casted =  cast_list_as_strings(list(df["question1"]))
    q2_casted =  cast_list_as_strings(list(df["question2"]))
    
    ############### Begin exercise ###################
    # what is kaggle                  q1
    # What is the kaggle platform     q2
    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)    
    X_q1q2 = scipy.sparse.hstack((X_q1,X_q2))
    ############### End exercise ###################

    return X_q1q2

def get_mistakes(clf, X_q1q2, y):
    ############### Begin exercise ###################
    predictions = clf.predict(X_q1q2)
    incorrect_predictions = predictions != y 
    incorrect_indices,  = np.where(incorrect_predictions)
    ############### End exercise ###################
    
    if np.sum(incorrect_predictions)==0:
        print("no mistakes in this df")
    else:
        return incorrect_indices, predictions

# ADDED ─────────────────────────────────────────────────────────────────────
# Extra imports required by the added functions
import pickle          # ADDED
import os              # ADDED
import sklearn.metrics # ADDED
from collections import Counter  # ADDED

# ADDED ─────────────────────────────────────────────────────────────────────
# Section 1: From-scratch similarity / distance metrics
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_similarity(str1, str2):  # ADDED
    """
    Jaccard similarity between the word-token sets of two strings.
    J(A, B) = |A ∩ B| / |A ∪ B|
    Implemented from scratch.
    """
    tokens1 = set(str(str1).lower().split())  # ADDED
    tokens2 = set(str(str2).lower().split())  # ADDED
    union = tokens1 | tokens2  # ADDED
    if not union:  # ADDED
        return 0.0  # ADDED
    return len(tokens1 & tokens2) / len(union)  # ADDED


def char_bigram_dice(str1, str2):  # ADDED
    """
    Sørensen–Dice coefficient on character-bigram multisets.
    D(A, B) = 2|A ∩ B| / (|A| + |B|)
    Robust to minor spelling differences; implemented from scratch.
    """
    def _bigrams(s):  # ADDED
        s = str(s).lower()  # ADDED
        return [s[i:i + 2] for i in range(len(s) - 1)]  # ADDED

    bg1 = _bigrams(str1)  # ADDED
    bg2 = _bigrams(str2)  # ADDED
    if not bg1 and not bg2:  # ADDED
        return 1.0  # ADDED
    if not bg1 or not bg2:  # ADDED
        return 0.0  # ADDED
    c1, c2 = Counter(bg1), Counter(bg2)  # ADDED
    intersection = sum((c1 & c2).values())  # ADDED
    return 2 * intersection / (len(bg1) + len(bg2))  # ADDED


def lcs_ratio(str1, str2):  # ADDED
    """
    Word-level Longest Common Subsequence (LCS) ratio, implemented from scratch.
    Returns len(LCS) / max(|words1|, |words2|).
    Complexity: O(m·n) per pair — included for demonstration; use on small batches.
    """
    words1 = str(str1).lower().split()  # ADDED
    words2 = str(str2).lower().split()  # ADDED
    m, n = len(words1), len(words2)  # ADDED
    if max(m, n) == 0:  # ADDED
        return 0.0  # ADDED
    dp = [[0] * (n + 1) for _ in range(m + 1)]  # ADDED
    for i in range(1, m + 1):  # ADDED
        for j in range(1, n + 1):  # ADDED
            if words1[i - 1] == words2[j - 1]:  # ADDED
                dp[i][j] = dp[i - 1][j - 1] + 1  # ADDED
            else:  # ADDED
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # ADDED
    return dp[m][n] / max(m, n)  # ADDED


def cosine_similarity_tfidf_batch(df, tfidf_vectorizer):  # ADDED
    """
    Pair-wise cosine similarity between TF-IDF vectors of question1 and question2.
    The cosine formula is applied from scratch on the sparse matrices:
        cos(u, v) = (u · v) / (||u|| · ||v||)
    Fully vectorised — no Python loops.
    Returns a (n_samples, 1) numpy array.
    """
    q1_list = cast_list_as_strings(list(df["question1"]))  # ADDED
    q2_list = cast_list_as_strings(list(df["question2"]))  # ADDED
    X_q1 = tfidf_vectorizer.transform(q1_list)  # ADDED
    X_q2 = tfidf_vectorizer.transform(q2_list)  # ADDED
    # Element-wise product summed per row → dot product per pair  # ADDED
    dot_products = np.array(X_q1.multiply(X_q2).sum(axis=1)).flatten()  # ADDED
    norms_q1 = np.sqrt(np.array(X_q1.power(2).sum(axis=1)).flatten())   # ADDED
    norms_q2 = np.sqrt(np.array(X_q2.power(2).sum(axis=1)).flatten())   # ADDED
    denom = norms_q1 * norms_q2  # ADDED
    cosine_sims = np.where(denom > 0, dot_products / denom, 0.0)  # ADDED
    return cosine_sims.reshape(-1, 1)  # ADDED


# ADDED ─────────────────────────────────────────────────────────────────────
# Section 2: Composite feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_handcrafted_features(df, tfidf_vectorizer):  # ADDED
    """
    Dense feature matrix (n_samples × 5) of handcrafted similarity signals:
      col 0 – Jaccard similarity        (from scratch)
      col 1 – Length ratio              min_len / max_len
      col 2 – Common-word F1 ratio      (from scratch)
      col 3 – Char-bigram Dice          (from scratch)
      col 4 – TF-IDF cosine similarity  (from scratch, vectorised)
    """
    q1_list = cast_list_as_strings(list(df["question1"]))  # ADDED
    q2_list = cast_list_as_strings(list(df["question2"]))  # ADDED
    n = len(q1_list)  # ADDED
    feats = np.zeros((n, 5), dtype=np.float32)  # ADDED

    for i, (q1, q2) in enumerate(zip(q1_list, q2_list)):  # ADDED
        w1 = set(q1.lower().split())  # ADDED
        w2 = set(q2.lower().split())  # ADDED

        # col 0 – Jaccard  # ADDED
        union = w1 | w2  # ADDED
        feats[i, 0] = len(w1 & w2) / len(union) if union else 0.0  # ADDED

        # col 1 – length ratio  # ADDED
        l1, l2 = len(q1.split()), len(q2.split())  # ADDED
        feats[i, 1] = min(l1, l2) / (max(l1, l2) + 1e-9)  # ADDED

        # col 2 – common-word F1  # ADDED
        common = len(w1 & w2)  # ADDED
        feats[i, 2] = 2 * common / (len(w1) + len(w2) + 1e-9)  # ADDED

        # col 3 – character bigram Dice  # ADDED
        feats[i, 3] = char_bigram_dice(q1, q2)  # ADDED

    # col 4 – TF-IDF cosine (vectorised, no loop)  # ADDED
    feats[:, 4] = cosine_similarity_tfidf_batch(df, tfidf_vectorizer).flatten()  # ADDED

    return feats  # ADDED


def get_combined_features(df, count_vectorizer, tfidf_vectorizer):  # ADDED
    """
    Concatenate sparse BoW features with dense handcrafted features into a
    single scipy sparse matrix.
    """
    X_bow  = get_features_from_df(df, count_vectorizer)              # ADDED
    X_hand = get_handcrafted_features(df, tfidf_vectorizer)          # ADDED
    X_combined = scipy.sparse.hstack(                                 # ADDED
        (X_bow, scipy.sparse.csr_matrix(X_hand))                     # ADDED
    )                                                                 # ADDED
    return X_combined                                                 # ADDED


# ADDED ─────────────────────────────────────────────────────────────────────
# Section 3: Model persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_object(obj, filepath):  # ADDED
    """Serialise *obj* to *filepath* using pickle."""
    with open(filepath, "wb") as f:  # ADDED
        pickle.dump(obj, f)           # ADDED


def load_object(filepath):  # ADDED
    """Deserialise and return the object stored at *filepath*."""
    with open(filepath, "rb") as f:  # ADDED
        return pickle.load(f)         # ADDED


# ADDED ─────────────────────────────────────────────────────────────────────
# Section 4: Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(clf, X, y, model_name="model", split_name="split"):  # ADDED
    """
    Compute ROC-AUC, Precision, Recall and F1 for a fitted classifier.

    Parameters
    ----------
    clf        : fitted sklearn classifier with predict_proba
    X          : feature matrix (sparse or dense)
    y          : true binary labels (numpy array)
    model_name : string label for the model
    split_name : 'train', 'val', or 'test'

    Returns
    -------
    dict with keys: model, split, roc_auc, precision, recall, f1
    """
    y_pred  = clf.predict(X)                   # ADDED
    y_proba = clf.predict_proba(X)[:, 1]       # ADDED
    return {                                    # ADDED
        "model":     model_name,                # ADDED
        "split":     split_name,                # ADDED
        "roc_auc":   round(sklearn.metrics.roc_auc_score(y, y_proba), 4),                      # ADDED
        "precision": round(sklearn.metrics.precision_score(y, y_pred, zero_division=0), 4),    # ADDED
        "recall":    round(sklearn.metrics.recall_score(y, y_pred, zero_division=0), 4),       # ADDED
        "f1":        round(sklearn.metrics.f1_score(y, y_pred, zero_division=0), 4),           # ADDED
    }                                           # ADDED