from __future__ import annotations
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union
import json, re, shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict 
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import torch
from tqdm import tqdm
import csv

# --- Global Configuration ---
MODEL_ID = "dicta-il/dictabert-tiny-joint"
BATCH_SIZE = 32  # Analyze 32 sentences at once.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Resource Lists ---
GERUND_NO_SUFFIX_SET: Set[str] = set()
GERUND_WITH_SUFFIX_SET: Set[str] = set()
PATH_TO_NO_SUFFIX = "gerunds_no_suffix.txt"
PATH_TO_WITH_SUFFIX = "gerunds_with_suffix.txt"

# --- Global State ---
_tokenizer = None
_model = None

# --- Paths ---
InputBibalRelativePath = "InputFiles/bibal"
InputMishnaRelativePath = "InputFiles/hazal/mishna"
InputRambamRelativePath = "InputFiles/hazal/rambam"
InputModernRelativePath = "InputFiles/modern"
JsonFilesBibalRelativePath = "JsonFiles/bibal"
JsonFilesMishnaRelativePath = "JsonFiles/hazal/mishna"
JsonFilesRambamRelativePath = "JsonFiles/hazal/rambam"
JsonFilesModernRelativePath = "JsonFiles/modern"

# --- Statistics Configuration ---
wordsFor_stat_2 = [
    "ילד", "תינוק", "שופט", "דיין", "שפה", "לשון", "סיר", "קדרה",
    "צמד", "זוג", "אהבה", "חיבה", "בטן", "כרס", "גם", "אף",
    "גבול", "תחום", "עם", "אומה", "ריב", "קטטה", "סיבה", "עילה"
]
wordsFor_stat_3 = ["ש", "אשר", "כי", "כאשר", "מאשר", "אם", "פן"]
CSV_EXPORT_DATA = []

# ==========================================
#              CORE LOGIC
# ==========================================

def _load_model():
    """
    Loads the model onto the GPU (CUDA) with FP16 precision for maximum speed.
    """
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"Loading Model on {DEVICE.upper()}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # Load model with trust_remote_code=True
        _model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            do_lex=False,
            do_syntax=True,
            do_ner=True,
            do_prefix=False,
            do_morph=True,
        )
        # MOVE TO GPU
        _model.to(DEVICE)
        if DEVICE == "cuda":
            _model.half() # Use half precision (much faster)
        _model.eval()
        print("Model loaded successfully.")
    return _tokenizer, _model

def load_gerund_sets():
    """ Loads the gerund text files into the global sets if they aren't loaded yet. """
    global GERUND_NO_SUFFIX_SET, GERUND_WITH_SUFFIX_SET
    if not GERUND_NO_SUFFIX_SET and Path(PATH_TO_NO_SUFFIX).exists():
        with open(PATH_TO_NO_SUFFIX, "r", encoding="utf-8") as f:
            GERUND_NO_SUFFIX_SET = set(line.strip() for line in f if line.strip())    
    if not GERUND_WITH_SUFFIX_SET and Path(PATH_TO_WITH_SUFFIX).exists():
        with open(PATH_TO_WITH_SUFFIX, "r", encoding="utf-8") as f:
            GERUND_WITH_SUFFIX_SET = set(line.strip() for line in f if line.strip())

def analyzeText(text_input: Union[str, List[str]], output_style: str = "json") -> Any:
    """
    Analyzes text using batches to maximize GPU usage.
    """
    tokenizer, model = _load_model()
    # Case 1: Single String
    if isinstance(text_input, str):
        # Wrap in list and process
        text_input = [text_input]
    # Case 2: List of Strings (Batch Processing)
    if isinstance(text_input, list):
        all_results = []
        # Disable gradient calculation for inference (Save Memory & Time)
        with torch.no_grad():
            # Process in chunks of BATCH_SIZE
            for i in tqdm(range(0, len(text_input), BATCH_SIZE), desc="Batching", leave=False):
                batch = text_input[i : i + BATCH_SIZE]
                # DictaBERT's custom predict method handles tokenization internally
                # We just pass the list of strings
                batch_results = model.predict(batch, tokenizer, output_style=output_style)
                all_results.extend(batch_results)
        return all_results
    return []

def sliceText(text: str, allow_colon: bool = False) -> List[str]:
    """
    Splits text into individual sentences.
    allow_colon: Set to True for Bible files where ':' ends a verse.
    """
    if allow_colon:
        # Split on Period, Exclamation, Question, Sof Pasuq, AND Colon
        pattern = r'([.!?:;]|\u05C3)\s*'
    else:
        # Split on Period, Exclamation, Question, Sof Pasuq ONLY
        # (Standard Modern/Rabbinic splitting)
        pattern = r'([.!?]|\u05C3)\s*'
    splitSentences = re.split(pattern, text)
    sentences = [splitSentences[2*i].strip() + splitSentences[2*i+1] for i in range(0, len(splitSentences)//2)]
    if len(splitSentences) % 2 != 0:
        val = splitSentences[-1].strip()
        if val: sentences.append(val)
    return [s for s in sentences if len(s.split()) > 1]

def iterate_folder_slice_text(sampleText, func_vars, file):
    outputFolder = func_vars[0]
    # Check if this file is from the Bible folder
    # "bibal" is in your variable InputBibalRelativePath
    is_biblical = "bibal" in file.parts or "bibal" in str(file.parent).lower()
    # Pass the flag
    sentences = sliceText(sampleText, allow_colon=is_biblical)
    if not sentences: return
    print(f"Analyzing {file.name} ({len(sentences)} sentences)...")
    result_list = analyzeText(sentences) 
    save_json(result_list, outputFolder, file.name)

def iterateFolder(inputFolder, func, func_vars: list):
    inputFolderPath = Path(inputFolder)
    # Check if we are saving files (func_vars[0] is a path string)
    # or just counting stats (func_vars[0] is a Counter/Dict)
    is_saving_files = len(func_vars) > 0 and isinstance(func_vars[0], (str, Path))
    files = sorted([f for f in inputFolderPath.rglob("*") if f.is_file() and not f.name.startswith(".")])
    for file in tqdm(files, desc=f"Processing {inputFolderPath.name}"):
        current_vars = func_vars
        # LOGIC FOR SAVING FILES (Subfolders)
        if is_saving_files:
            outputRoot = Path(func_vars[0])
            relative_parent = file.relative_to(inputFolderPath).parent
            target_output_folder = outputRoot / relative_parent
            target_output_folder.mkdir(parents=True, exist_ok=True)
            # Update the first argument to be the specific subfolder
            current_vars = [str(target_output_folder)] + func_vars[1:]
        if file.suffix.lower() == ".xml":
            sampleText = readXmlText(file)
        else:
            try:
                sampleText = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    sampleText = file.read_text(encoding="cp1255")
                except:
                    continue
        if sampleText.strip():
            func(sampleText, current_vars, file)

def iterate_folder_count_scentences_length(sampleText, func_vars, file):
    output_dict = func_vars[0]  
    # Check if this file is Biblical
    is_biblical = "bibal" in file.parts or "bibal" in str(file.parent).lower()
    if is_biblical:
        pattern = r'([.!?:;]|\u05C3)\s*'
    else:
        pattern = r'([.!?]|\u05C3)\s*'
    splitSentences = re.split(pattern, sampleText)
    sentences = [splitSentences[2*i].strip() + splitSentences[2*i+1] for i in range(0, len(splitSentences)//2)] 
    if len(splitSentences) % 2 != 0:
        val = splitSentences[-1].strip()
        if val: sentences.append(val)   
    for sentence in sentences:
        words = [w for w in sentence.split(" ") if w.strip()]
        if words:
            output_dict[len(words)] += 1

def readXmlText(path: Path) -> str:
    try:
        root = ET.parse(path).getroot()
        all_text = " ".join([t.strip() for t in root.itertext() if t and t.strip()]) 
        filtered_tokens = []
        has_started_hebrew = False
        for w in all_text.split():
            # 1. Check if word contains a Hebrew letter
            is_hebrew = any('\u0590' <= c <= '\u05ff' for c in w)
            # 2. Check if it is a punctuation mark we care about
            is_punct = all(c in '.!?:;\u05C3-,\"' for c in w)
            if is_hebrew:
                has_started_hebrew = True
                filtered_tokens.append(w)
            # Keep punctuation ONLY if we have already started seeing Hebrew.
            # This prevents us from keeping the periods from the English header!
            elif is_punct and has_started_hebrew:
                filtered_tokens.append(w)          
        return " ".join(filtered_tokens)
    except: return ""

def save_json(obj: Any, base_dir: str, filename: str):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    jsonName = filename.split(".")[0] + ".json"
    path = base / jsonName
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=None) # indent=None saves disk space!

# ==========================================
#              STATISTICS
# ==========================================

def countWords(folder_path: str, wordCounter, target_list: list):
    folder = Path(folder_path)
    json_files = list(folder.rglob("*.json"))
    for filePath in tqdm(json_files, desc=f"Counting {folder.name}", leave=False):
        try:
            with open(filePath, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict): data = [data]
            for sentence_obj in data:
                tokens = sentence_obj.get("tokens", [])
                for token in tokens:
                    word = ""
                    # Navigate down the path (e.g. ["morph", "pos"])
                    current_obj = token
                    valid = True
                    for target in target_list:
                        if isinstance(current_obj, dict):
                            current_obj = current_obj.get(target)
                        else:
                            valid = False; break
                    if valid and current_obj is not None:
                        # If we are counting specific words (target_list=["token"] or ["lex"])
                        if target_list == ["token"] or target_list == ["lex"]:
                            text_val = str(current_obj)
                            # Keep only if it has at least one Hebrew letter
                            if not any('\u0590' <= c <= '\u05ff' for c in text_val):
                                continue
                        if isinstance(current_obj, dict):
                            for k, v in current_obj.items(): wordCounter[k][v] += 1
                        else:
                            wordCounter[current_obj] += 1
        except Exception as e: pass

def group_into_bins(data_dict: dict[int, int], bin_size: int = 5, max_val: int = 60) -> dict:
    """
    Groups numerical counts into bins.
    """
    total_items = sum(data_dict.values())
    if total_items == 0: return {}
    bins = {}
    for i in range(0, max_val + 1, bin_size):
        label = f"{i}-{i+(bin_size-1)}" if i < max_val else f"{max_val}+"
        bins[label] = 0
    for val, count in data_dict.items():
        if val >= max_val:
            bins[f"{max_val}+"] += count
        else:
            group_start = (val // bin_size) * bin_size
            label = f"{group_start}-{group_start+(bin_size-1)}"
            if label in bins:
                bins[label] += count
    return {k: float(v) / total_items for k, v in bins.items()}

def build_dependency_tree(tokens: list) -> tuple[dict, list]:
    """
    Builds a dependency tree from a list of tokens.
    Returns:
        children_map: dict mapping parent_idx -> list of child indices.
        roots: list of root token indices (dep_head_idx == -1).
    """
    children_map = defaultdict(list)
    roots = []
    for i, token in enumerate(tokens):
        head_idx = token.get("syntax", {}).get("dep_head_idx", -1)
        if head_idx == -1:
            roots.append(i)
        else:
            children_map[head_idx].append(i)
    return children_map, roots

def extract_all_json_stats(json_folder_path: str, stats: dict):
    """
    Single-pass processor: Reads each JSON file exactly once and extracts
    data for Unique Words, Syntactic Depth, Word Order, Gerunds, and Possessives.
    """
    load_gerund_sets() 
    folder = Path(json_folder_path)
    files = list(folder.rglob("*.json"))
    for file_path in tqdm(files, desc=f"Extracting Stats from {folder.name}", leave=False):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f) 
            if isinstance(data, dict): data = [data]
            for sentence in data:
                tokens = sentence.get("tokens", [])
                if len(tokens) <= 1:
                    continue
                children_map, roots = build_dependency_tree(tokens)
                pos_map = {i: t.get("morph", {}).get("pos", "") for i, t in enumerate(tokens)}
                unique_lexemes = set()
                def max_depth(node, visited): #STAT 6: Syntactic Depth
                    if node in visited: return 0
                    visited.add(node)
                    if not children_map[node]:
                        visited.remove(node)
                        return 0
                    depths = [max_depth(child, visited) for child in children_map[node]]
                    visited.remove(node)
                    return 1 + max(depths) if depths else 0
                if roots:
                    depth = max(max_depth(r, set()) for r in roots)
                    stats["depth"][depth] += 1
                # --- TOKEN-LEVEL LOOP ---
                for i, token in enumerate(tokens):
                    syntax = token.get("syntax", {})
                    morph = token.get("morph", {})
                    dep_func = syntax.get("dep_func", "")
                    head_idx = syntax.get("dep_head_idx", -1)
                    pos = morph.get("pos", "")
                    feats = morph.get("feats", {})
                    word_str = token.get("token", "").strip()
                    lex_str = morph.get("lex", word_str).strip()
                    if any('\u05d0' <= c <= '\u05ea' for c in lex_str): # Ensure it's a Hebrew word, not punctuation
                        unique_lexemes.add(lex_str) #STAT 5: Unique Words
                        stats["total_words"] += 1
                    if lex_str in wordsFor_stat_2: #STAT 2: Edith Doron Words (Filtered)
                        if lex_str == "עם" and pos != "NOUN":
                            pass # Skip it, it means "with"
                        else:
                            stats["doron_words"][lex_str] += 1
                    if dep_func == "nsubj" and head_idx != -1: #STAT 8b: Word Order (V1 vs V2)
                        if pos_map.get(head_idx) == "VERB":
                            if i < head_idx:
                                stats["word_order"]["SV (V2 Tendency)"] += 1
                            elif i > head_idx:
                                stats["word_order"]["VS (V1 Tendency)"] += 1
                    has_nsubj = any( #STAT 8c: Infinitive vs Gerund
                        tokens[child_idx].get("syntax", {}).get("dep_func", "") == "nsubj" 
                        for child_idx in children_map.get(i, [])
                    )
                    is_infinitive = (pos == "VERB") and (not feats) and (not has_nsubj)
                    in_no_suffix = (lex_str in GERUND_NO_SUFFIX_SET) or (word_str in GERUND_NO_SUFFIX_SET)
                    in_with_suffix = (lex_str in GERUND_WITH_SUFFIX_SET) or (word_str in GERUND_WITH_SUFFIX_SET)
                    is_gerund = (
                        (in_no_suffix and has_nsubj) or 
                        in_with_suffix or 
                        ((pos == "VERB") and (not feats) and has_nsubj)
                    )
                    if is_infinitive: stats["inf_gerund"]["Infinitive"] += 1
                    if is_gerund: stats["inf_gerund"]["Gerund"] += 1
                    if dep_func == "compound:smixut": #STAT 8d: Possessives (Smixut vs Shel)
                        stats["possessives"]["Construct-State (Smixut)"] += 1
                    elif lex_str in ["של", "שלי", "שלו", "שלה", "שלנו", "שלכם", "שלכן", "שלהם", "שלהן"]:
                        stats["possessives"]["Prepositional (Shel)"] += 1
                # Update the Unique Words sentence-level counter
                num_unique = len(unique_lexemes)
                if num_unique > 0:
                    stats["unique"][num_unique] += 1
        except Exception: 
            pass

def stat_1(wordDict: dict[str, int]) -> dict:
    num = sum(wordDict.values())
    return {k: float(v)/num for k, v in wordDict.items()} if num else {}

def stat_2(doron_dict: dict[str, int], total_words: int) -> dict:
    if total_words == 0: return {}
    return {k: float(doron_dict.get(k, 0)) / total_words for k in wordsFor_stat_2 if k in doron_dict}

def stat_3(wordDict: dict[str, int]) -> dict:
    num = sum(wordDict.values())
    return {k: float(wordDict[k])/num for k in wordsFor_stat_3 if k in wordDict} if num else {}

def stat_4(lengthDict: dict[int, int]) -> dict:
    # Sentence Length: Bins of 5, max 60
    return group_into_bins(lengthDict, bin_size=5, max_val=60)

def stat_5(uniqueDict: dict[int, int]) -> dict:
    # Unique Words: Bins of 5, max 60
    return group_into_bins(uniqueDict, bin_size=5, max_val=60)

def stat_6(depthDict: dict[int, int]) -> dict:
    # Syntactic Depth: Small bins of 2, max 14 (eliminates empty space)
    return group_into_bins(depthDict, bin_size=2, max_val=14)

def stat_7_8a(dir_path: str, path_to_objective) -> dict:
    word_dict = Counter()
    countWords(dir_path, word_dict, path_to_objective)
    return stat_1(word_dict)

def stat_7b(dir_path: str) -> List[Dict[str, float]]:
    word_dict: Dict[str, Counter] = {
        "Gender": Counter(), "Number": Counter(), "Person": Counter(), "Tense": Counter()
    }
    countWords(dir_path, word_dict, ["morph", "feats"])
    results = []
    for k in ["Gender", "Number", "Person", "Tense"]:
        total = sum(word_dict[k].values())
        results.append({sub_k: v/total for sub_k, v in word_dict[k].items()} if total else {})
    return results

def stat_8b(orderDict: dict[str, int]) -> dict:
    """ Word Order (SV vs VS) percentage distribution """
    total = sum(orderDict.values())
    if total == 0: return {}
    return {k: float(v) / total for k, v in orderDict.items()}

def stat_8c(verbDict: dict[str, int]) -> dict:
    """ Infinitive vs Gerund percentage distribution """
    total = sum(verbDict.values())
    if total == 0: return {}
    return {k: float(v) / total for k, v in verbDict.items()}

def stat_8d(possDict: dict[str, int]) -> dict:
    """ Possessive construction percentage distribution """
    total = sum(possDict.values())
    if total == 0: return {}
    return {k: float(v) / total for k, v in possDict.items()}

def print_7b_stats():
    paths = [JsonFilesBibalRelativePath, JsonFilesMishnaRelativePath, JsonFilesRambamRelativePath, JsonFilesModernRelativePath]
    names = ["Bibal", "Mishna", "Rambam", "Modern"]
    all_stats = [stat_7b(p) for p in paths] # List of [Gender, Number, Person, Tense] per period
    features = ["Gender", "Number", "Person", "Tense"]
    for i, feature in enumerate(features):
        data_list = []
        for period_idx, period_name in enumerate(names):
            data_list.append((period_name, all_stats[period_idx][i]))
        save_tower_graph(f"{feature} Feature Distribution", f"{feature}", "Frequency", data_list)

def save_tower_graph(title, x_label, y_label, values_list: list[tuple[str, dict]], out_dir="plots", ext="png"):
    all_keys = set().union(*(d.keys() for _, d in values_list))
    def sort_key(k):
        s = str(k)
        nums = re.findall(r'\d+', s)
        if nums:
            return int(nums[0])
        return s 
    keys = sorted([k for k in all_keys if str(k).strip()], key=sort_key)
    if not keys: return
    x_pos = np.arange(len(keys))
    width = 0.8 / max(len(values_list), 1)
    plt.figure(figsize=(15, 8))
    for i, (name, d) in enumerate(values_list):
        plt.bar(x_pos + i * width, [d.get(k, 0) for k in keys], width, label=name)
    centers = x_pos + width * (len(values_list) - 1) / 2
    # Matplotlib prints Hebrew backwards. This reverses the string if it contains Hebrew letters.
    def fix_hebrew(text):
        return text[::-1] if any('\u05d0' <= c <= '\u05ea' for c in text) else text
    clean_labels = [fix_hebrew(re.sub(r'[$_^{}\\]', '', str(k))) for k in keys]
    rotation = 90 if len(keys) > 25 else 45
    plt.xticks(centers, clean_labels, rotation=rotation, ha="right", fontsize=12) # X-axis values
    plt.yticks(fontsize=12) # Y-axis values
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=12)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try: 
        plt.tight_layout()
    except: 
        plt.subplots_adjust(bottom=0.30, right=0.85) # Increased bottom margin to prevent label cutoff   
    plt.savefig(out / f"{title}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()

def generate_similarity_report(stat_name: str, stat_dicts: list[dict], names: list[str]):
    """ Generates terminal output AND saves data to the global CSV list. """
    print(f"\n{'='*95}")
    print(f" SIMILARITY ANALYSIS: {stat_name.upper()}")
    print(f"{'='*95}")
    metrics = ["Manhattan", "Euclidean", "Cosine", "Jaccard", "Dice", "JS Div"]
    row_format = "{:<20} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}"
    print(row_format.format("Comparison Pairs", *metrics))
    print("-" * 95)
    CSV_EXPORT_DATA.append([]) # Blank row for spacing
    CSV_EXPORT_DATA.append([f"SIMILARITY ANALYSIS: {stat_name.upper()}"])
    CSV_EXPORT_DATA.append(["Comparison Pairs"] + metrics)
    pairs = [(3, 0), (3, 1), (3, 2), (0, 1), (0, 2)]
    for idx1, idx2 in pairs:
        name1, name2 = names[idx1], names[idx2]
        v1, v2 = align_vectors(stat_dicts[idx1], stat_dicts[idx2])
        manhattan = calc_manhattan_distance(v1, v2)
        euclidean = calc_euclidean_distance(v1, v2)
        cosine = calc_cosine_similarity(v1, v2)
        jaccard = calc_jaccard_similarity(v1, v2)
        dice = calc_dice_similarity(v1, v2)
        jsd = calc_jensen_shannon(v1, v2)
        pair_name = f"{name1} vs {name2}"
        print(row_format.format(
            pair_name, 
            f"{manhattan:.4f}", f"{euclidean:.4f}", f"{cosine:.4f}", 
            f"{jaccard:.4f}", f"{dice:.4f}", f"{jsd:.4f}"
        ))
        CSV_EXPORT_DATA.append([
            pair_name, 
            f"{manhattan:.4f}", f"{euclidean:.4f}", f"{cosine:.4f}", 
            f"{jaccard:.4f}", f"{dice:.4f}", f"{jsd:.4f}"
        ])

def printStatistics():
    print("Generating Statistics...")
    word_dicts = [Counter() for _ in range(4)]
    length_dicts = [Counter() for _ in range(4)]
    unique_dicts = [Counter() for _ in range(4)]
    depth_dicts = [Counter() for _ in range(4)]
    word_order_dicts = [Counter() for _ in range(4)]
    inf_gerund_dicts = [Counter() for _ in range(4)]
    poss_dicts = [Counter() for _ in range(4)]
    doron_dicts = [Counter() for _ in range(4)]
    total_words_counts = [0 for _ in range(4)]
    paths = [
        ("Bible", InputBibalRelativePath, JsonFilesBibalRelativePath), 
        ("Mishna", InputMishnaRelativePath, JsonFilesMishnaRelativePath),
        ("Rambam", InputRambamRelativePath, JsonFilesRambamRelativePath), 
        ("Modern", InputModernRelativePath, JsonFilesModernRelativePath)
    ]
    for i, (name, inPath, jsonPath) in enumerate(paths):
        print(f"--- {name} ---")
        countWords(jsonPath, word_dicts[i], ["token"]) # 1. Simple Word Counts
        iterateFolder(inPath, iterate_folder_count_scentences_length, [length_dicts[i]]) # 2. Sentence Lengths (Raw Text)
        stats_bundle = { # 3. Master JSON Pass
            "unique": unique_dicts[i],
            "depth": depth_dicts[i],
            "word_order": word_order_dicts[i],
            "inf_gerund": inf_gerund_dicts[i],
            "possessives": poss_dicts[i],
            "doron_words": doron_dicts[i],  
            "total_words": 0                
        }
        extract_all_json_stats(jsonPath, stats_bundle)
        # Save the total word count for this period after the extraction finishes
        total_words_counts[i] = stats_bundle["total_words"]
    print("Plotting...")
    all_words = Counter()
    for d in word_dicts: all_words.update(d)
    top_50_keys = []
    for word, count in all_words.most_common():
        clean_word = str(word).strip()
        if len(clean_word) > 1: 
            top_50_keys.append(clean_word)
        if len(top_50_keys) >= 50:
            break
    filtered_word_stats = []
    for i, p in enumerate(paths):
        filtered_d = {k: v for k, v in word_dicts[i].items() if k in top_50_keys}
        stat = stat_1(filtered_d) 
        filtered_word_stats.append((p[0], stat))

    # --- Plotting all graphs ---
    save_tower_graph("Words Frequency (Top 50)", "Word", "Frequency", filtered_word_stats)
    save_tower_graph("Edith Doron Special Words", "Word", "Frequency", [(p[0], stat_2(doron_dicts[i], total_words_counts[i])) for i, p in enumerate(paths)])
    save_tower_graph("Subordinating conjunctions", "Word", "Frequency", [(p[0], stat_3(word_dicts[i])) for i, p in enumerate(paths)])
    save_tower_graph("Sentence Length distribution", "Sentence Length", "Frequency", [(p[0], stat_4(length_dicts[i])) for i, p in enumerate(paths)])
    save_tower_graph("Unique Words Per Sentence", "Unique Word Count", "Frequency", [(p[0], stat_5(unique_dicts[i])) for i, p in enumerate(paths)])
    save_tower_graph("Syntactic Tree Depth", "Depth of Sentence Tree", "Frequency", [(p[0], stat_6(depth_dicts[i])) for i, p in enumerate(paths)])
    save_tower_graph("Clausal Word Order (SV vs VS)", "Order", "Frequency", [(p[0], stat_8b(word_order_dicts[i])) for i, p in enumerate(paths)])
    save_tower_graph("Infinitive vs Gerund Distribution", "Type", "Frequency", [(p[0], stat_8c(inf_gerund_dicts[i])) for i, p in enumerate(paths)])
    save_tower_graph("Possessive Constructions (Smixut vs Shel)", "Type", "Frequency", [(p[0], stat_8d(poss_dicts[i])) for i, p in enumerate(paths)])
    pos_stats = [(p[0], stat_7_8a(p[2], ["morph", "pos"])) for p in paths]
    save_tower_graph("POS Category Distribution", "Category", "Frequency", pos_stats)
    dep_stats = [(p[0], stat_7_8a(p[2], ["syntax", "dep_func"])) for p in paths]
    save_tower_graph("Syntactic Relations", "Relation Type", "Frequency", dep_stats)
    print_7b_stats()

    # ==========================================
    #   FULL SIMILARITY / DISTANCE REPORTS
    # ==========================================
    print("\n\n" + "#"*70)
    print(" "*20 + "COMPREHENSIVE DISTANCE ANALYSIS")
    print("#"*70)
    names = [p[0] for p in paths] # ["Bibal", "Mishna", "Rambam", "Modern"]
    # Stat 1: Words Frequency (Top 50)
    stat1_dicts = [stat_1({k: v for k, v in word_dicts[i].items() if k in top_50_keys}) for i in range(4)]
    generate_similarity_report("Stat 1: Words Frequency (Top 50)", stat1_dicts, names)
    # Stat 2: Edith Doron Special Words
    stat2_dicts = [stat_2(doron_dicts[i], total_words_counts[i]) for i in range(4)]
    generate_similarity_report("Stat 2: Edith Doron Special Words", stat2_dicts, names)
    # Stat 3: Subordinating Conjunctions
    stat3_dicts = [stat_3(word_dicts[i]) for i in range(4)]
    generate_similarity_report("Stat 3: Subordinating Conjunctions", stat3_dicts, names)
    # Stat 4: Sentence Length Distribution
    stat4_dicts = [stat_4(length_dicts[i]) for i in range(4)]
    generate_similarity_report("Stat 4: Sentence Length Distribution", stat4_dicts, names)
    # Stat 5: Unique Words Per Sentence
    stat5_dicts = [stat_5(unique_dicts[i]) for i in range(4)]
    generate_similarity_report("Stat 5: Unique Words Per Sentence", stat5_dicts, names)
    # Stat 6: Syntactic Tree Depth
    stat6_dicts = [stat_6(depth_dicts[i]) for i in range(4)]
    generate_similarity_report("Stat 6: Syntactic Tree Depth", stat6_dicts, names)
    # Stat 7/8a: POS Category Distribution
    stat7_pos_dicts = [stat_7_8a(p[2], ["morph", "pos"]) for p in paths]
    generate_similarity_report("Stat 7 & 8a: POS Category Distribution", stat7_pos_dicts, names)
    # Stat 7/8a: Syntactic Relations
    stat7_dep_dicts = [stat_7_8a(p[2], ["syntax", "dep_func"]) for p in paths]
    generate_similarity_report("Stat 7 & 8a: Syntactic Relations", stat7_dep_dicts, names)
    # Stat 8b: Clausal Word Order (SV vs VS)
    stat8b_dicts = [stat_8b(word_order_dicts[i]) for i in range(4)]
    generate_similarity_report("Stat 8b: Clausal Word Order (SV vs VS)", stat8b_dicts, names)
    # Stat 8c: Infinitive vs Gerund Distribution
    stat8c_dicts = [stat_8c(inf_gerund_dicts[i]) for i in range(4)]
    generate_similarity_report("Stat 8c: Infinitive vs Gerund", stat8c_dicts, names)
    # Stat 8d: Possessive Constructions (Smixut vs Shel)
    stat8d_dicts = [stat_8d(poss_dicts[i]) for i in range(4)]
    generate_similarity_report("Stat 8d: Possessive Constructions", stat8d_dicts, names)

    with open("Linguistic_Distance_Reports.csv", mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(CSV_EXPORT_DATA)

    print("Done.")

# ==========================================
#              VECTOR SIMILARITY METRICS
# ==========================================

def align_vectors(dict1: dict, dict2: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns two dictionaries into numpy arrays of the same length.
    Fills missing keys with 0.0.
    """
    all_keys = set(dict1.keys()).union(set(dict2.keys()))
    # Sort keys to ensure vectors are strictly aligned
    sorted_keys = sorted(list(all_keys), key=str) 
    vec1 = np.array([dict1.get(k, 0.0) for k in sorted_keys])
    vec2 = np.array([dict2.get(k, 0.0) for k in sorted_keys])
    return vec1, vec2

def calc_manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """ Equation 9: Manhattan Distance """
    return float(np.sum(np.abs(v1 - v2)))

def calc_euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """ Equation 10: Euclidean Distance """
    return float(np.sqrt(np.sum((v1 - v2)**2)))

def calc_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """ Equation 11: Cosine Similarity """
    if np.sum(v1) == 0 or np.sum(v2) == 0: return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def calc_jaccard_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """ Equation 13: Weighted Jaccard Similarity """
    den = np.sum(np.maximum(v1, v2))
    if den == 0: return 0.0
    return float(np.sum(np.minimum(v1, v2)) / den)

def calc_dice_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """ Equation 15: Weighted Dice Similarity """
    den = np.sum(v1 + v2)
    if den == 0: return 0.0
    return float((2 * np.sum(np.minimum(v1, v2))) / den)

def calc_jensen_shannon(v1: np.ndarray, v2: np.ndarray) -> float:
    """ 
    Alternative to Eq 16 (KL Divergence): Jensen-Shannon Divergence.
    The paper notes KL Divergence fails when Q(x)=0. JS Divergence fixes this.
    Returns distance (0 is identical, 1 is completely different).
    """
    # JS expects probability distributions (sum to 1)
    s1, s2 = np.sum(v1), np.sum(v2)
    if s1 == 0 or s2 == 0: return 1.0
    p = v1 / s1
    q = v2 / s2
    return float(distance.jensenshannon(p, q))

# ==========================================
#              MAIN
# ==========================================
def inpufFile(filePath: str, folderPath: str):
    source = Path(filePath)
    dest = Path(folderPath)
    dest.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, dest / source.name)
    else:
        print("File not found.")

def runAnalysis():
    print("Starting Analysis (CUDA Optimized)...")
    iterateFolder(InputBibalRelativePath, iterate_folder_slice_text, [JsonFilesBibalRelativePath])
    iterateFolder(InputMishnaRelativePath, iterate_folder_slice_text, [JsonFilesMishnaRelativePath])
    iterateFolder(InputRambamRelativePath, iterate_folder_slice_text, [JsonFilesRambamRelativePath])
    iterateFolder(InputModernRelativePath, iterate_folder_slice_text, [JsonFilesModernRelativePath])            

def buildDirectoryPath():
    for p in [InputBibalRelativePath, InputMishnaRelativePath, InputRambamRelativePath, InputModernRelativePath,
            JsonFilesBibalRelativePath, JsonFilesMishnaRelativePath, JsonFilesRambamRelativePath, JsonFilesModernRelativePath]:
        Path(p).mkdir(parents=True, exist_ok=True)

def main():
    buildDirectoryPath()
    menu = [
        ("Load File", lambda: inpufFile(input("Source: "), input("Dest: "))),
        ("Run Analysis", runAnalysis),
        ("Print Statistics", printStatistics),
        ("Exit", None)
    ]
    while True:
        print("\n=== MENU ===")
        for i, (cmd, _) in enumerate(menu): print(f"{i} - {cmd}")
        choice = input("Select: ")
        if choice.isdigit() and int(choice) < len(menu):
            if int(choice) == len(menu) - 1: break
            menu[int(choice)][1]()

if __name__ == "__main__":
    main()