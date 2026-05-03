import numpy as np
import scipy
import re
import sklearn
import sklearn.metrics

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

# Extra imports required by the added functions
import pickle          
import os              
from collections import Counter  

# Section 1: From-scratch similarity / distance metrics
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_similarity(str1, str2):  
    """
    Jaccard similarity between the word-token sets of two strings.
    J(A, B) = |A ∩ B| / |A ∪ B|
    Implemented from scratch.
    """
    tokens1 = set(str(str1).lower().split())  
    tokens2 = set(str(str2).lower().split())  
    union = tokens1 | tokens2  
    if not union:  
        return 0.0  
    return len(tokens1 & tokens2) / len(union)  


def char_bigram_dice(str1, str2):  
    """
    Sørensen–Dice coefficient on character-bigram multisets.
    D(A, B) = 2|A ∩ B| / (|A| + |B|)
    Robust to minor spelling differences; implemented from scratch.
    """
    def _bigrams(s):  
        s = str(s).lower()  
        return [s[i:i + 2] for i in range(len(s) - 1)]  

    bg1 = _bigrams(str1)  
    bg2 = _bigrams(str2)  
    if not bg1 and not bg2:  
        return 1.0  
    if not bg1 or not bg2:  
        return 0.0  
    c1, c2 = Counter(bg1), Counter(bg2)  
    intersection = sum((c1 & c2).values())  
    return 2 * intersection / (len(bg1) + len(bg2))  


def lcs_ratio(str1, str2):  
    """
    Word-level Longest Common Subsequence (LCS) ratio, implemented from scratch.
    Returns len(LCS) / max(|words1|, |words2|).
    Complexity: O(m·n) per pair — included for demonstration; use on small batches.
    """
    words1 = str(str1).lower().split()  
    words2 = str(str2).lower().split()  
    m, n = len(words1), len(words2)  
    if max(m, n) == 0:  
        return 0.0  
    dp = [[0] * (n + 1) for _ in range(m + 1)]  
    for i in range(1, m + 1):  
        for j in range(1, n + 1):  
            if words1[i - 1] == words2[j - 1]:  
                dp[i][j] = dp[i - 1][j - 1] + 1  
            else:  
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  
    return dp[m][n] / max(m, n)  


def cosine_similarity_tfidf_batch(df, tfidf_vectorizer):  
    """
    Pair-wise cosine similarity between TF-IDF vectors of question1 and question2.
    The cosine formula is applied from scratch on the sparse matrices:
        cos(u, v) = (u · v) / (||u|| · ||v||)
    Fully vectorised — no Python loops.
    Returns a (n_samples, 1) numpy array.
    """
    q1_list = cast_list_as_strings(list(df["question1"]))  
    q2_list = cast_list_as_strings(list(df["question2"]))  
    X_q1 = tfidf_vectorizer.transform(q1_list)  
    X_q2 = tfidf_vectorizer.transform(q2_list)  
    # Element-wise product summed per row → dot product per pair  
    dot_products = np.array(X_q1.multiply(X_q2).sum(axis=1)).flatten()  
    norms_q1 = np.sqrt(np.array(X_q1.power(2).sum(axis=1)).flatten())   
    norms_q2 = np.sqrt(np.array(X_q2.power(2).sum(axis=1)).flatten())   
    denom = norms_q1 * norms_q2  

    # cosine_sims = np.where(denom > 0, dot_products / denom, 0.0)  
    cosine_sims = np.zeros_like(dot_products)
    np.divide(dot_products, denom, out=cosine_sims, where=denom > 0)
    return cosine_sims.reshape(-1, 1)  

def get_handcrafted_features(df, tfidf_vectorizer):  
    """
    Dense feature matrix (n_samples × 5) of handcrafted similarity signals:
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
    q1_list = cast_list_as_strings(list(df["question1"]))  
    q2_list = cast_list_as_strings(list(df["question2"]))  
    n = len(q1_list)  
    feats = np.zeros((n, 9), dtype=np.float32)  

    # Pre-compilation to extract words without puntuation signs
    word_pattern = re.compile(r'\b\w+\b')

    for i, (q1, q2) in enumerate(zip(q1_list, q2_list)):  
        w1 = set(q1.lower().split())  
        w2 = set(q2.lower().split())  

        # col 0 – Jaccard  
        union = w1 | w2  
        feats[i, 0] = len(w1 & w2) / len(union) if union else 0.0  

        # col 1 – length ratio  
        l1, l2 = len(q1.split()), len(q2.split())  
        feats[i, 1] = min(l1, l2) / (max(l1, l2) + 1e-9)  

        # col 2 – common-word F1  
        common = len(w1 & w2)  
        feats[i, 2] = 2 * common / (len(w1) + len(w2) + 1e-9)  

        # col 3 – character bigram Dice  
        feats[i, 3] = char_bigram_dice(q1, q2)  

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

    # col 8 – TF-IDF cosine (vectorised, no loop)  
    feats[:, 8] = cosine_similarity_tfidf_batch(df, tfidf_vectorizer).flatten()  

    return feats  


def get_combined_features(df, count_vectorizer, tfidf_vectorizer):  
    """
    Concatenate sparse BoW features with dense handcrafted features into a
    single scipy sparse matrix.
    """
    X_bow  = get_interaction_features_from_df(df, count_vectorizer) # Using count_vectorizer works better than using tf-idf
    X_hand = get_handcrafted_features(df.copy(), tfidf_vectorizer)          
    X_combined = scipy.sparse.hstack(                                 
        (X_bow, scipy.sparse.csr_matrix(X_hand))                     
    )                                                                 
    return X_combined                                                 


# Section 3: Model persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_object(obj, filepath):  
    """Serialise *obj* to *filepath* using pickle."""
    with open(filepath, "wb") as f:  
        pickle.dump(obj, f)           


def load_object(filepath):  
    """Deserialise and return the object stored at *filepath*."""
    with open(filepath, "rb") as f:  
        return pickle.load(f)         


# Section 4: Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(clf, X, y, model_name="model", split_name="split"):  
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
    y_pred  = clf.predict(X)                   
    y_proba = clf.predict_proba(X)[:, 1]       
    return {                                    
        "model":     model_name,                
        "split":     split_name,                
        "roc_auc":   round(sklearn.metrics.roc_auc_score(y, y_proba), 4),                      
        "precision": round(sklearn.metrics.precision_score(y, y_pred, zero_division=0), 4),    
        "recall":    round(sklearn.metrics.recall_score(y, y_pred, zero_division=0), 4),       
        "f1":        round(sklearn.metrics.f1_score(y, y_pred, zero_division=0), 4),           
        "accuracy":  round(sklearn.metrics.accuracy_score(y, y_pred), 4),
    }                                           

# Section 5: SBERT (Sentence-BERT) embedding features
# SBERT encodes each question independently into a dense vector (O(n) inference).
# paraphrase-MiniLM-L6-v2 is fine-tuned specifically on paraphrase detection,
# which maps directly to the Quora task. See utils_Arnau_Soler.ipynb for details.

def get_sbert_interaction_features(df, sbert_model):  
    """
    Encode question pairs with SBERT and return a (n, 2*dim) interaction matrix.
    First half: |emb_q1 - emb_q2|  (what is different between the two questions).
    Second half: emb_q1 * emb_q2   (what is shared between the two questions).
    Mirrors the design of get_interaction_features_from_df but in semantic space.
    """
    q1_list = cast_list_as_strings(list(df["question1"]))  
    q2_list = cast_list_as_strings(list(df["question2"]))  

    # --- Encode all question1 strings ----------------------------------------
    # batch_size=256 amortises the Transformer overhead across many sentences
    # at once; show_progress_bar gives visible feedback on the large train set
    emb_q1 = sbert_model.encode(   
        q1_list,                    
        batch_size=256,             
        show_progress_bar=True,     
        convert_to_numpy=True,      
    )                               

    # --- Encode all question2 strings ----------------------------------------
    emb_q2 = sbert_model.encode(   
        q2_list,                    
        batch_size=256,             
        show_progress_bar=True,     
        convert_to_numpy=True,      
    )                               

    # --- Build interaction features (mirrors get_interaction_features_from_df) ---

    # Absolute difference: |emb_q1 − emb_q2|  — shape (n, dim)
    # A duplicate pair should produce small differences across all dimensions
    diff = np.abs(emb_q1 - emb_q2)   

    # Element-wise product: emb_q1 ⊙ emb_q2  — shape (n, dim)
    # A duplicate pair should produce large positive values in shared directions
    prod = emb_q1 * emb_q2            

    # Concatenate horizontally → (n_samples, 2 * embedding_dim)
    return np.hstack([diff, prod]).astype(np.float32)  


# Section 6: Graph / "Magic" features
# The Quora dataset forms a graph: questions are nodes, CSV rows are edges.
# Two structural signals proved the strongest in top Kaggle solutions:
#   1. High-degree nodes (frequent questions) tend to be duplicates.
#   2. Questions sharing ≥1 common neighbour are far less likely to be duplicates.
# build_freq_dict and build_neighbor_dict MUST be called on training data only.
# See utils_Arnau_Soler.ipynb for the full analysis.

def build_freq_dict(df):  
    """
    Count how many times each question appears across both columns of df.
    Equivalent to the node degree in the question-pair graph.
    Must be called on training data only to avoid leakage.
    Returns dict {question_string: int}.
    """
    freq_dict = {}  
    all_questions = (                                          
        cast_list_as_strings(list(df["question1"])) +         
        cast_list_as_strings(list(df["question2"]))           
    )                                                          
    for q in all_questions:                                    
        freq_dict[q] = freq_dict.get(q, 0) + 1               
    return freq_dict                                           


def build_neighbor_dict(df):  
    """
    Build the adjacency list of the question-pair graph from df.
    neighbor_dict[q] = set of all questions q is paired with.
    Must be called on training data only to avoid leakage.
    Returns dict {question_string: set of neighbor strings}.
    """
    neighbor_dict = {}                                                  
    q1_list = cast_list_as_strings(list(df["question1"]))              
    q2_list = cast_list_as_strings(list(df["question2"]))              
    for q1, q2 in zip(q1_list, q2_list):                               
        # Add q2 to q1's neighbor set and vice-versa (undirected graph)
        neighbor_dict.setdefault(q1, set()).add(q2)                    
        neighbor_dict.setdefault(q2, set()).add(q1)                    
    return neighbor_dict                                                


def get_graph_features(df, freq_dict, neighbor_dict):  
    """
    Extract 6 graph-based features per question pair (n_samples, 6):
      col 0 – q1_freq   : how often q1 appears in training data
      col 1 – q2_freq   : how often q2 appears in training data
      col 2 – freq_min  : min(q1_freq, q2_freq)
      col 3 – freq_max  : max(q1_freq, q2_freq)
      col 4 – freq_diff : |q1_freq - q2_freq|
      col 5 – intersect : |neighbors(q1) ∩ neighbors(q2)|  (the key "magic" feature)
    Unseen questions get freq=0 and empty neighbor set.
    """
    q1_list = cast_list_as_strings(list(df["question1"]))    
    q2_list = cast_list_as_strings(list(df["question2"]))    
    n = len(q1_list)                                          
    feats = np.zeros((n, 6), dtype=np.float32)               

    for i, (q1, q2) in enumerate(zip(q1_list, q2_list)):    
        f1 = freq_dict.get(q1, 0)                            
        f2 = freq_dict.get(q2, 0)                            

        # col 0-4: frequency-based features
        feats[i, 0] = f1                                     
        feats[i, 1] = f2                                     
        feats[i, 2] = min(f1, f2)                            
        feats[i, 3] = max(f1, f2)                            
        feats[i, 4] = abs(f1 - f2)                           

        # col 5: neighbor intersection — the "magic" feature
        # Set intersection is O(min(|N1|,|N2|)); most nodes have small degree
        # so this is fast in practice even for 300 k pairs.
        n1 = neighbor_dict.get(q1, set())                    
        n2 = neighbor_dict.get(q2, set())                    
        feats[i, 5] = len(n1 & n2)                           

    return feats                                              


def get_sbert_graph_features(df, sbert_model,                
                              freq_dict, neighbor_dict,       
                              sbert_feat_path=None):          
    """
    Combine SBERT interaction features with graph features into a single matrix.
    Layout: [|emb_q1-emb_q2| (384) | emb_q1*emb_q2 (384) | graph (6)] = (n, 774).
    If sbert_feat_path exists the SBERT block is loaded from disk instead of
    re-encoding, which saves time when train_models.ipynb has already run.
    """
    # --- Load or compute the SBERT block ------------------------------------
    if sbert_feat_path is not None and os.path.exists(sbert_feat_path):  
        X_sbert = np.load(sbert_feat_path)                               
    else:                                                                  
        X_sbert = get_sbert_interaction_features(df, sbert_model)        

    # --- Compute the graph block --------------------------------------------
    X_graph = get_graph_features(df, freq_dict, neighbor_dict)            

    # --- Concatenate and return ---------------------------------------------
    return np.hstack([X_sbert, X_graph]).astype(np.float32)              