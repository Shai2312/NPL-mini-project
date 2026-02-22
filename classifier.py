import os
import json
import random
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# IMPORT DICTA-BERT FROM MAIN
try:
    from main import analyzeText
except ImportError:
    analyzeText = None
    print("WARNING: 'main.py' not found. Manual input will not work.")

# ==========================================
#              CONFIGURATION
# ==========================================
PATH_BIBAL = "JsonFiles/bibal"
PATH_MISHNA = "JsonFiles/hazal/mishna"
PATH_RAMBAM = "JsonFiles/hazal/rambam"
PATH_MODERN = "JsonFiles/modern"
MODERN_SUB_GENRES = ["blogs", "literary", "medical", "news", "tapuz"]
# Resources for Linguistic Mode
LIST_DORON_LEXICAL = [
    "ילד", "תינוק", "שופט", "דיין", "שפה", "לשון", "סיר", "קדרה", 
    "צמד", "זוג", "אהבה", "חיבה", "בטן", "כרס", "גם", "אף", 
    "גבול", "תחום", "עם", "אומה", "ריב", "קטטה", "סיבה", "עילה"
]
LIST_CONJUNCTIONS = ["ש", "אשר", "כי", "כאשר", "מאשר", "אם", "פן"]
try:
    GERUND_WITH_SUFFIX = set(open("gerunds_with_suffix.txt", encoding="utf-8").read().splitlines())
except: GERUND_WITH_SUFFIX = set()

# ==========================================
#        1. LINGUISTIC CALCULATORS 
# ==========================================

def get_tree_depth(tokens):
    if not tokens: return 0
    children = {i: [] for i in range(len(tokens))}
    roots = []
    for i, t in enumerate(tokens):
        # Safe access using .get()
        h = t.get("syntax", {}).get("dep_head_idx")
        if h is None or h == -1 or h == i: roots.append(i)
        else: children.setdefault(h, []).append(i)
    # Safety against infinite recursion if there are loops in the tree
    try:
        def depth(n, visited=set()):
            if n in visited: return 0 
            visited.add(n)
            return 1 + max((depth(c, visited.copy()) for c in children[n]), default=0)
        return max((depth(r) for r in roots), default=0)
    except RecursionError:
        return 0

def calc_v1_v2(tokens):
    root_idx = -1
    for i, t in enumerate(tokens):
        syn = t.get("syntax", {})
        if (syn.get("dep_head_idx") == -1 or syn.get("dep_func") == "ROOT") and \
           t.get("morph", {}).get("pos") == "VERB":
            root_idx = i; break
    if root_idx == -1: return 0.0
    for i in range(root_idx):
        t = tokens[i]
        pos = t.get("morph", {}).get("pos")
        lex = t.get("lex") or t.get("token")
        if pos in ["CCONJ", "SCONJ", "PUNCT"]: continue
        if lex in ["ו", "ש", "כי", "אם"]: continue
        if t.get("syntax", {}).get("dep_head_idx") == root_idx: return 0.0 
    return 1.0 

def calc_gerunds_infinitives(tokens):
    gerunds, infinitives = 0, 0
    children = {i: [] for i in range(len(tokens))}
    for i, t in enumerate(tokens):
        h = t.get("syntax", {}).get("dep_head_idx", -1)
        if h != -1: children.setdefault(h, []).append(i)
    for i, t in enumerate(tokens):
        morph = t.get("morph", {})
        pos = morph.get("pos")
        token_str = t.get("token", "")
        # Safe Check for Subject in children
        has_subj = False
        for c in children[i]:
            child_syn = tokens[c].get("syntax", {})
            if child_syn.get("dep_func") == "nsubj":
                has_subj = True
                break
        if pos == "VERB" and not morph.get("feats") and not has_subj:
            infinitives += 1
        elif (pos == "VERB" and has_subj) or (token_str in GERUND_WITH_SUFFIX):
            gerunds += 1
    return gerunds, infinitives

def calc_possessives(tokens):
    construct, shel = 0, 0
    for t in tokens:
        syn = t.get("syntax", {})
        if syn.get("dep_func") == "compound:smixut": construct += 1
        lex = t.get("lex") or ""
        token_str = t.get("token") or ""
        if lex == "של" or token_str == "של": shel += 1
    return construct, shel

def calc_doron_keywords(tokens):
    count_lex, count_conj = 0, 0
    for t in tokens:
        w = t.get("lex") or t.get("token")
        if w in LIST_DORON_LEXICAL: count_lex += 1
        if w in LIST_CONJUNCTIONS: count_conj += 1
    return count_lex, count_conj

# ==========================================
#        2. FEATURE EXTRACTION
# ==========================================

def tokens_to_features(tokens, mode):
    """Converts JSON tokens to the format required by the classifier."""
    if not tokens: return None
    # --- MODE 1: LEXICAL (The words themselves) ---
    if mode == 'lexical':
        return " ".join([t.get('lex') or t.get('token') for t in tokens])
    # --- MODE 2: MORPHOLOGY (Inflection Patterns) ---
    elif mode == 'morph':
        feats = []
        for t in tokens:
            morph = t.get('morph', {})
            pos = morph.get('pos', 'X')
            extra = morph.get('feats', {})
            # Sorted to ensure 'Gen=M|Num=S' is same as 'Num=S|Gen=M'
            sorted_feats = sorted([f"{k}={v}" for k, v in extra.items() if k in ['Gender', 'Number', 'Person', 'Tense']])
            feat_str = "|".join(sorted_feats)
            feats.append(f"{pos}__{feat_str}")
        return " ".join(feats)
    # --- MODE 3: SYNTAX (Dependency Edges) ---
    elif mode == 'syntax':
        rels = []
        for i, t in enumerate(tokens):
            syn = t.get('syntax', {})
            func = syn.get('dep_func', 'root')
            head = syn.get('dep_head_idx', -1)
            dist = 0 if head == -1 else head - i
            rels.append(f"{func}:{dist}")
        return " ".join(rels)
    # --- MODE 4: STRUCTURE (Abstract POS Sequence) ---
    elif mode == 'structure':
        return " ".join([t.get('morph', {}).get('pos', 'X') for t in tokens])
    # --- MODE 5: LINGUISTIC (Edith Doron & Stats) ---
    elif mode == 'linguistic':
        v1_score = calc_v1_v2(tokens)
        n_gerund, n_inf = calc_gerunds_infinitives(tokens)
        n_const, n_shel = calc_possessives(tokens)
        n_doron, n_conj = calc_doron_keywords(tokens)
        depth = get_tree_depth(tokens)
        words = [
            t.get('lex') for t in tokens 
            if t.get('lex') and any('\u05d0' <= c <= '\u05ea' for c in t.get('lex'))
        ]
        uniq_ratio = len(set(words)) / len(words) if words else 0
        return {
            "is_v1": v1_score,
            "n_gerund": n_gerund,
            "n_infinitive": n_inf,
            "n_construct": n_const,
            "n_shel": n_shel,
            "n_doron_lex": n_doron,
            "n_sub_conj": n_conj,
            "tree_depth": depth,
            "unique_ratio": uniq_ratio,
            "length": len(tokens)
        }
    return None

def extract_features_from_file(json_path, mode):
    """ Reads the nested JSON and returns a LIST of features (one per sentence). """
    features_list = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f) 
        # Handle both single dicts and lists of dicts
        if isinstance(data, dict): 
            data = [data]
        for sentence in data:
            tokens = sentence.get('tokens', [])
            # Filter out empty or 1-word garbage sentences
            if len(tokens) > 1:
                feat = tokens_to_features(tokens, mode)
                if feat:
                    features_list.append(feat)
        return features_list
    except Exception as e: 
        return []

# ==========================================
#        3. DATA LOADING 
# ==========================================

def load_training_data(feature_mode):
    X_bib, y_bib = [], []
    X_rab, y_rab = [], []
    print(f"Loading Training Data ({feature_mode})...")
    # 1. Biblical (Recursive)
    files_bib = glob(os.path.join(PATH_BIBAL, "**", "*.json"), recursive=True)
    for p in files_bib:
        feats = extract_features_from_file(p, feature_mode)
        if feats: 
            X_bib.extend(feats)               
            y_bib.extend([0] * len(feats))    # <-- Create labels for all sentences
    # 2. Rabbinic (Recursive Mishna + Rambam)
    files_rab = glob(os.path.join(PATH_MISHNA, "**", "*.json"), recursive=True) + \
                glob(os.path.join(PATH_RAMBAM, "**", "*.json"), recursive=True)
    for p in files_rab:
        feats = extract_features_from_file(p, feature_mode)
        if feats: 
            X_rab.extend(feats)               
            y_rab.extend([1] * len(feats))
    if not X_bib or not X_rab:
        print(f"  [CRITICAL] Missing data for mode '{feature_mode}'. Skipping training.")
        return [], np.array([])
    # 3. OVERSAMPLING BIBLICAL
    ratio = 1
    if len(X_bib) > 0 and len(X_rab) > len(X_bib):
        ratio = len(X_rab) // len(X_bib)
    if ratio > 1:
        print(f"  > Oversampling: Multiplying Biblical data by {ratio}...")
        X_bib = X_bib * ratio
        y_bib = y_bib * ratio
    # 4. Combine & Shuffle
    X = X_bib + X_rab
    y = y_bib + y_rab
    combined = list(zip(X, y))
    random.seed(42)
    random.shuffle(combined)
    X_final, y_final = zip(*combined)
    print(f"  > Final Training Set: {len(X_final)} sentences (Balanced).")
    return list(X_final), np.array(y_final)

# ==========================================
#        4. CLASSIFIER PIPELINE
# ==========================================

def get_pipeline(feature_mode):
    # Numeric Features
    if feature_mode == 'linguistic':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000))
        ])
    else:
        # Use Bigrams/Trigrams for Structure/Morph to capture context
        ngram = (1, 3) if feature_mode in ['structure', 'morph'] else (1, 1)
        return Pipeline([
            ('vect', TfidfVectorizer(ngram_range=ngram, min_df=5)),
            ('clf', LogisticRegression(max_iter=1000))
        ])

def train_and_evaluate(feature_mode):
    X_train, y_train = load_training_data(feature_mode)
    if not X_train: return None
    if feature_mode == 'linguistic':
        X_data = pd.DataFrame(X_train)
    else:
        X_data = np.array(X_train, dtype=object)
    pipeline = get_pipeline(feature_mode)
    print(f"  Running 10-Fold CV...")
    try:
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
        scores = cross_validate(pipeline, X_data, y_train, cv=10, scoring=scoring_metrics)
        print(f"  > Accuracy:  {scores['test_accuracy'].mean():.3f}")
        print(f"  > Precision: {scores['test_precision'].mean():.3f}")
        print(f"  > Recall:    {scores['test_recall'].mean():.3f}")
        print(f"  > F-Measure: {scores['test_f1'].mean():.3f}")
    except Exception as e:
        print(f"  [ERROR] CV Failed: {e}")
    pipeline.fit(X_data, y_train)
    return pipeline

# ==========================================
#        5. RUNNERS
# ==========================================

def run_modern_analysis(models):
    expected_modes = ["lexical", "morph", "syntax", "structure", "linguistic"]
    print("\n" + "="*100)
    print(f"{'MODERN HEBREW ANALYSIS IN PROGRESS...':^100}")
    print("="*100)
    final_table_rows = []
    for genre in MODERN_SUB_GENRES:
        genre_path = os.path.join(PATH_MODERN, genre)
        if os.path.exists(genre_path):
            files = list(Path(genre_path).rglob("*.json"))
        else:
            files = []
        # --- PROGRESS INDICATOR ---
        print(f"Extracting features from '{genre}' ({len(files)} files). Please wait...")
        results = []
        for mode in expected_modes:
            if mode not in models or models[mode] is None:
                results.append(f"{'NOT TRAINED':<12}")
                continue
            if not files:
                results.append(f"{'---':<12}")
                continue
            X_mod = []
            for p in files:
                feats = extract_features_from_file(str(p), mode)
                if feats: X_mod.extend(feats) 
            if not X_mod: 
                results.append(f"{'N/A':<12}")
                continue
            data = pd.DataFrame(X_mod) if mode == 'linguistic' else X_mod
            try:
                probs = models[mode].predict_proba(data)[:, 1] # Class 1 = Rabbinic
                avg_prob = probs.mean()
                if avg_prob > 0.5:
                    lbl = "Rab"
                    pct = int(avg_prob * 100)
                else:
                    lbl = "Bib"
                    pct = int((1 - avg_prob) * 100)
                results.append(f"{lbl} {pct}%".ljust(12))
            except:
                results.append(f"{'ERROR':<12}")
        row_str = f"| {genre:<12} | {len(files):<6} | " + " | ".join(results) + " |"
        final_table_rows.append(row_str)
    print("\n" + "="*100)
    print(f"{'FINAL PREDICTION RESULTS':^100}")
    print(f"{'(Predictions: 0=Biblical, 1=Rabbinic)':^100}")
    print("="*100)
    header = f"| {'Genre':<12} | {'Files':<6} | " + " | ".join([f"{m.upper():<12}" for m in expected_modes]) + " |"
    print(header)
    print("-" * len(header))
    for row in final_table_rows:
        print(row)
    print("-" * len(header))

def run_manual_input(models):
    if analyzeText is None: return
    print("\n" + "="*50 + "\n      MANUAL INPUT MODE (Type 'EXIT' to quit)\n" + "="*50)
    while True:
        text = input("\nEnter Hebrew Sentence: ").strip()
        if text.upper() == 'EXIT': break
        if not text: continue
        try:
            json_res = analyzeText(text)
            # 1. Handle both dicts and lists safely
            if isinstance(json_res, dict):
                json_res = [json_res]
            print(f"\n--- Analysis: {text} ---")
            # 2. Iterate through every sentence DictaBERT found
            for i, sentence_data in enumerate(json_res):
                tokens = sentence_data.get('tokens', [])
                # Skip garbage sentences (1 word or empty)
                if len(tokens) <= 1: continue 
                # If there are multiple sentences, label them
                if len(json_res) > 1:
                    print(f"\n[Sentence {i+1} Prediction]")
                for mode, pipeline in models.items():
                    if not pipeline: continue
                    feat = tokens_to_features(tokens, mode)
                    if not feat: continue
                    data = pd.DataFrame([feat]) if mode == 'linguistic' else [feat]
                    prob_rab = pipeline.predict_proba(data)[0][1]
                    label = "Rabbinic" if prob_rab > 0.5 else "Biblical"
                    conf = prob_rab if prob_rab > 0.5 else 1 - prob_rab
                    print(f"[{mode.upper():<10}] -> {label:<10} (Conf: {conf:.2f})")
        except Exception as e: 
            print(f"Error: {e}")

# ==========================================
#        MAIN
# ==========================================
if __name__ == "__main__":
    # ALL MODES ENABLED
    modes = ["lexical", "morph", "syntax", "structure", "linguistic"]
    trained_models = {}
    print("--- 1. Training Classifiers ---")
    for mode in modes:
        trained_models[mode] = train_and_evaluate(mode)
    while True:
        print("\n--- MENU ---")
        print("1. Analyze Modern Genres")
        print("2. Manual Input")
        print("3. Exit")
        ch = input("Select: ")
        if ch == '1': run_modern_analysis(trained_models)
        elif ch == '2': run_manual_input(trained_models)
        elif ch == '3': break