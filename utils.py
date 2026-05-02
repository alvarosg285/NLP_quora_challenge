import numpy as np
import scipy
import re

def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings

def get_interaction_features_from_df(df, count_vectorizer):
    """
    Returns a sparse matrix containing the interaction features built by the count vectorizer.
    Instead of horizontally stacking the features of question1 and question2 independently, 
    it computes the absolute difference and the element-wise product of their sparse matrices.
    This provides symmetric features and allows linear models to capture specific word overlaps.
    """
    q1_casted = cast_list_as_strings(list(df["question1"]))
    q2_casted = cast_list_as_strings(list(df["question2"]))

    # Transform text into sparse Bag-of-Words matrices
    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)    
    
    # 1. Absolute Difference: |X_q1 - X_q2|
    # If a word is present in both, 1 - 1 = 0
    # If a word is present in only one of them, |1 - 0| = 1 or |0 - 1| = 1
    # In this way, we expect the linear model to learn negative weights for words that do not match
    X_diff = abs(X_q1 - X_q2)
    
    # 2. Element-wise Product: X_q1 * X_q2
    # If a word is present in both, 1 * 1 = 1
    # If it is missing in one of them, 1 * 0 = 0
    # Like this, we want the linear model to learn which specific shared words are key to predict duplicates
    X_prod = X_q1.multiply(X_q2)
    
    # Horizontally stack the interaction matrices instead of the independent vectors
    X_interactions = scipy.sparse.hstack((X_diff, X_prod))

    return X_interactions

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

def print_mistake_k(k, df, mistake_indices, predictions):
    print(df.iloc[mistake_indices[k]].question1)
    print(df.iloc[mistake_indices[k]].question2)
    print("true class:", df.iloc[mistake_indices[k]].is_duplicate)
    print("prediction:", predictions[mistake_indices[k]])
    print()

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

    # cosine_sims = np.where(denom > 0, dot_products / denom, 0.0)  # ADDED
    cosine_sims = np.zeros_like(dot_products)
    np.divide(dot_products, denom, out=cosine_sims, where=denom > 0)
    return cosine_sims.reshape(-1, 1)  # ADDED


# ADDED ─────────────────────────────────────────────────────────────────────
# Section 2: Composite feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_handcrafted_features(df, tfidf_vectorizer):  # ADDED
    """
    Dense feature matrix (n_samples × 9) of handcrafted similarity signals:
      col 0 – Jaccard similarity        (from scratch)
      col 1 – Length ratio              min_len / max_len
      col 2 – Common-word F1 ratio      (from scratch)
      col 3 – Char-bigram Dice          (from scratch)
      col 4 – Number mismatch           (1 if one question has a number and the other one hasn't)
      col 5 – Exact numbers match       (1 if both questions have at least a number and the numbers are the same)
      col 6 – Uppercase match           (1 if both questions share the same words that start with an uppercase)
      col 7 – First word match          (1 if both questions start with the same word)
      col 8 – TF-IDF cosine similarity  (from scratch, vectorised)
    """
    q1_list = cast_list_as_strings(list(df["question1"]))  # ADDED
    q2_list = cast_list_as_strings(list(df["question2"]))  # ADDED
    n = len(q1_list)  # ADDED
    feats = np.zeros((n, 9), dtype=np.float32)  # ADDED

    # Pre-compilation to extract words without puntuation signs
    word_pattern = re.compile(r'\b\w+\b')

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

        # FEATURES REGARDING THE NUMBERS THAT THE QUESITONS CONTAIN
        nums1 = re.findall(r'\d+', q1)
        nums2 = re.findall(r'\d+', q2)

        has_num1 = len(nums1) > 0
        has_num2 = len(nums2) > 0

        # col 4 – Number mismatch
        feats[i, 4] = 1.0 if has_num1 != has_num2 else 0.0

        # col 5 – Exact numbers match
        # We use sorted() so that ['1', '2'] is equal to ['2', '1']
        if has_num1 and has_num2 and sorted(nums1) == sorted(nums2):
            feats[i, 5] = 1.0
        else:
            feats[i, 5] = 0.0
        
        # FEATURE REGARDING UPPERCASES AND POSITION ---
        # We keep uppercases here
        words1 = word_pattern.findall(q1)
        words2 = word_pattern.findall(q2)

        # We ignore the first word and we keep only the words beginning with an uppercase
        caps1 = {w for w in words1[1:] if w[0].isupper()} if len(words1) > 1 else set()
        caps2 = {w for w in words2[1:] if w[0].isupper()} if len(words2) > 1 else set()

        # col 6 – Uppercase match
        if len(caps1) > 0 and len(caps2) > 0 and caps1 == caps2:
            feats[i, 6] = 1.0
        else:
            feats[i, 6] = 0.0
        
        # col 7 – First word match
        if len(words1) > 0 and len(words2) > 0:
            if words1[0].lower() == words2[0].lower():
                feats[i, 7] = 1.0
            else:
                feats[i, 7] = 0.0
        else:
            feats[i, 7] = 0.0

    # col 8 – TF-IDF cosine (vectorised, no loop)  # ADDED
    feats[:, 8] = cosine_similarity_tfidf_batch(df, tfidf_vectorizer).flatten()  # ADDED

    return feats  # ADDED


def get_combined_features(df, count_vectorizer, tfidf_vectorizer):  # ADDED
    """
    Concatenate sparse BoW features with dense handcrafted features into a
    single scipy sparse matrix.
    """
    X_bow  = get_interaction_features_from_df(df, count_vectorizer) # Using count_vectorizer works better than using tf-idf
    X_hand = get_handcrafted_features(df.copy(), tfidf_vectorizer)          # ADDED
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
    Compute ROC-AUC, Precision, Recall, F1 and Accuracy for a fitted classifier.

    Parameters
    ----------
    clf        : fitted sklearn classifier with predict_proba
    X          : feature matrix (sparse or dense)
    y          : true binary labels (numpy array)
    model_name : string label for the model
    split_name : 'train', 'val', or 'test'

    Returns
    -------
    dict with keys: model, split, roc_auc, precision, recall, f1, accuracy
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
        "accuracy":  round(sklearn.metrics.accuracy_score(y, y_pred), 4),
    }                                           # ADDED