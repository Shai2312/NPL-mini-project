from __future__ import annotations
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import List, Dict, Any, Set
import json, re, shutil
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm  # Progress bar library

# --- Global Configuration ---
MODEL_ID = "dicta-il/dictabert-tiny-joint"
MAX_NUM_OF_WORDS = 200

# --- Global State ---
_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Resource Lists ---
GERUND_NO_SUFFIX_SET: Set[str] = set()
GERUND_WITH_SUFFIX_SET: Set[str] = set()
PATH_TO_NO_SUFFIX = "gerunds_no_suffix.txt" 
PATH_TO_WITH_SUFFIX = "gerunds_with_suffix.txt"

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
    "ילד", "תינוק",   # Child / Infant
    "שופט", "דיין",   # Judge
    "שפה", "לשון",    # Language
    "סיר", "קדרה",    # Pot
    "צמד", "זוג",     # Pair
    "אהבה", "חיבה",   # Love / Affection
    "בטן", "כרס",     # Belly
    "גם", "אף",       # Also
    "גבול", "תחום",   # Limit / Delimitation
    "עם", "אומה",     # People / Nation
    "ריב", "קטטה",    # Feud / Brawl
    "סיבה", "עילה"    # Reason / Cause
]

wordsFor_stat_3 = ["ש", "אשר", "כי", "כאשר", "מאשר", "אם", "פן"]


# ==========================================
#              CORE LOGIC
# ==========================================

def load_resources():
    """Loads external word lists and the AI model into memory ONCE."""
    global GERUND_NO_SUFFIX_SET, GERUND_WITH_SUFFIX_SET, _tokenizer, _model
    
    # 1. Load Word Lists
    if not GERUND_NO_SUFFIX_SET:
        print("Loading gerund lists...")
        GERUND_NO_SUFFIX_SET = _load_word_list(PATH_TO_NO_SUFFIX)
        GERUND_WITH_SUFFIX_SET = _load_word_list(PATH_TO_WITH_SUFFIX)
        print(f"Loaded {len(GERUND_NO_SUFFIX_SET)} words (no suffix) and {len(GERUND_WITH_SUFFIX_SET)} words (with suffix).")

    # 2. Load Model (Force Single Load)
    if _tokenizer is None or _model is None:
        print(f"Loading DictaBERT model on {_device.upper()}...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            _model = AutoModel.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                do_lex=True,       
                do_syntax=True,    
                do_ner=True,
                do_prefix=False,
                do_morph=True,     
            )
            _model.to(_device) # Move to GPU if available
            _model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR LOADING MODEL: {e}")
            exit(1)

def _load_word_list(path_str: str) -> Set[str]:
    p = Path(path_str)
    if not p.exists():
        # Create empty files if missing to prevent crash
        print(f"Warning: File not found: {path_str}. Creating empty file.")
        p.touch()
        return set()
    try:
        content = p.read_text(encoding="utf-8")
        return {line.strip() for line in content.splitlines() if line.strip()}
    except Exception as e:
        print(f"Error reading {path_str}: {e}")
        return set()

def analyzeText(text: str, output_style: str = "json") -> Dict[str, Any]:
    global _tokenizer, _model
    
    if _model is None:
        # Emergency fallback
        load_resources()

    assert _model is not None

    # Model inference
    result = _model.predict([text], _tokenizer, output_style=output_style)
    
    if isinstance(result, list):
        return result[0] 
    return result

# ==========================================
#          STAT FUNCTIONS
# ==========================================
# (These remain unchanged, just condensed for brevity in this paste)

def stat_1(wordDict: dict[str, int]) -> dict:
    num = sum(wordDict.values())
    return {k: float(v)/num for k, v in wordDict.items()} if num else {}

def stat_2(wordDict: dict[str, int]) -> dict:
    num = sum(wordDict.values())
    return {k: float(wordDict[k])/num for k in wordsFor_stat_2 if k in wordDict} if num else {}

def stat_3(wordDict: dict[str, int]) -> dict:
    num = sum(wordDict.values())
    return {k: float(wordDict[k])/num for k in wordsFor_stat_3 if k in wordDict} if num else {}

def stat_4(lengthDict: dict[int, int]) -> dict:
    num = sum(lengthDict.values())
    return {k: float(lengthDict[k])/num for k in lengthDict} if num else {}

def stat_5(tokens: List[Dict[str, Any]]) -> int:
    unique_words = set()
    for token in tokens:
        word_form = token.get("lex") or token.get("token")
        if word_form: unique_words.add(word_form)
    return len(unique_words)

def stat_6(tokens: List[Dict[str, Any]]) -> int:
    if not tokens: return 0
    children = {i: [] for i in range(len(tokens))}
    roots = []
    for i, token in enumerate(tokens):
        head_idx = token.get("syntax", {}).get("dep_head_idx") 
        if head_idx is None or head_idx == -1: roots.append(i)
        elif head_idx != i: 
            if head_idx in children: children[head_idx].append(i)
            else: roots.append(i)
    
    def get_depth(node_idx):
        if not children[node_idx]: return 1 
        return 1 + max(get_depth(child) for child in children[node_idx])

    return max(get_depth(root) for root in roots) if roots else 0

def stat_7b(dir_path: str) -> list:
    word_dict = {"Gender": Counter(), "Number": Counter(), "Person": Counter(), "Tense": Counter()}
    countWords(dir_path, word_dict, ["morph", "feats"])
    results = []
    for feat in ["Gender", "Number", "Person", "Tense"]:
        total = sum(word_dict[feat].values())
        results.append({k: v/total for k, v in word_dict[feat].items()} if total else {})
    return results

def stat_7_8a(dir_path: str, path_to_objective) -> dict:
    word_dict = Counter()
    countWords(dir_path, word_dict, path_to_objective)
    return stat_1(word_dict)

def stat_8b(tokens: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    children_map = {i: [] for i in range(len(tokens))}
    for i, token in enumerate(tokens):
        head = token.get("syntax", {}).get("dep_head_idx")
        if head is not None and head != -1 and head != i:
            if head in children_map: children_map[head].append(i)

    results = {"infinitive": [], "gerund": []}
    for i, token in enumerate(tokens):
        word = token.get("token", "")
        lex = token.get("lex", "")
        morph = token.get("morph", {})
        pos = morph.get("pos", "")
        feats = morph.get("feats") or {} 
        
        has_nsubj = any(tokens[c].get("syntax", {}).get("dep_func") == "nsubj" for c in children_map[i])

        if pos == "VERB" and (not feats) and (not has_nsubj):
            results["infinitive"].append(word)
            continue 

        is_gerund = False
        if (word in GERUND_NO_SUFFIX_SET or lex in GERUND_NO_SUFFIX_SET) and has_nsubj: is_gerund = True
        elif (word in GERUND_WITH_SUFFIX_SET): is_gerund = True
        elif pos == "VERB" and has_nsubj: is_gerund = True
        
        if is_gerund: results["gerund"].append(word)
    return results

def stat_8c(tokens: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    results = {"construct_state": [], "prepositional_shel": []}
    for i, token in enumerate(tokens):
        syntax = token.get("syntax", {})
        dep_func = syntax.get("dep_func")
        head_idx = syntax.get("dep_head_idx")
        has_valid_head = (head_idx is not None and head_idx != -1 and 0 <= head_idx < len(tokens))

        if dep_func == "compound:smixut" and has_valid_head:
            results["construct_state"].append(f"{tokens[head_idx]['token']} {token['token']}")

        lex = token.get("lex")
        tok_text = token.get("token")
        if (lex == "של" or tok_text == "של") and dep_func == "case" and has_valid_head:
            possessor_head_idx = tokens[head_idx].get("syntax", {}).get("dep_head_idx")
            if possessor_head_idx is not None and possessor_head_idx != -1 and 0 <= possessor_head_idx < len(tokens):
                results["prepositional_shel"].append(f"{tokens[possessor_head_idx]['token']} {tok_text} {tokens[head_idx]['token']}")
    return results

# ==========================================
#           FILE PROCESSING (OPTIMIZED)
# ==========================================

def readXmlText(path: Path) -> str:
    root = ET.parse(path).getroot()
    parts = [text.strip() for text in root.itertext() if text.strip()]
    return " ".join(parts)

def sliceText(text: str) -> List[str]:
    splitSectences = re.split(r'([.!?])\s*', text)
    sentences = [splitSectences[2*i].strip() + splitSectences[2*i+1] for i in range(0, len(splitSectences)//2)]
    if len(splitSectences) % 2 != 0:
        val = splitSectences[-1].strip()
        if val: sentences.append(val)
    
    output = []
    currChunk = ""
    currSize = 0
    for sentence in sentences:
        words = sentence.split(" ")
        if currSize + len(words) <= MAX_NUM_OF_WORDS:
            currChunk += " " + sentence
            currSize += len(words)
        else:
            if currChunk: output.append(currChunk.strip())
            currChunk = sentence
            currSize = len(words)
    if currChunk: output.append(currChunk.strip())
    return output

def save_json(obj: Any, base_dir: str, filename: str):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    with (base / f"{filename.split('.')[0]}.json").open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)

def iterate_folder_slice_text(sampleText, func_vars, file):
    outputFolder = func_vars[0]
    textInSize = sliceText(sampleText)
    
    for index, text in enumerate(textInSize):
        if not text.strip(): continue
        result = analyzeText(text) # Uses loaded model
        save_json(result, outputFolder, f"{index}_{file.name}")

def iterate_folder_count_scentences_length(sampleText, func_vars, file):
    """
    Splits text into sentences and counts how many words are in each.
    Used for the 'Sentence Length distribution' graph.
    """
    output_dict = func_vars[0]
    
    # Split text by punctuation (. ! ?)
    splitSectences = re.split(r'([.!?])\s*', sampleText)
    
    # Reconstruct sentences (attach the punctuation back to the sentence)
    sentences = [splitSectences[2*i].strip() + splitSectences[2*i+1] for i in range(0, len(splitSectences)//2)]
    
    # Handle any remaining text at the end
    if len(splitSectences) % 2 != 0:
        val = splitSectences[-1].strip()
        if val: sentences.append(val)
    
    # Count words in each sentence and update the dictionary
    for sentence in sentences:
        # Filter out empty strings
        words = [w for w in sentence.split(" ") if w.strip()]
        if words:
            output_dict[len(words)] += 1

def iterateFolder(inputFolder, func, func_vars: list):
    inputFolderPath = Path(inputFolder)
    # Get total count for progress bar
    files = [f for f in inputFolderPath.rglob("*") if f.is_file() and not f.name.startswith(".")]
    
    if not files:
        print(f"No files found in {inputFolder}")
        return

    print(f"Processing {len(files)} files in {inputFolder}...")
    
    # TQDM Progress Bar
    for file in tqdm(files, desc="Analyzing", unit="file"):
        try:
            if file.suffix.lower() == ".xml":
                sampleText = readXmlText(file)
            else:
                try:
                    sampleText = file.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    sampleText = file.read_text(encoding="cp1255") # Hebrew fallback
            
            func(sampleText, func_vars, file)
        except Exception as e:
            tqdm.write(f"Error processing {file.name}: {e}")

def countWords(folder_path:str, wordCounter, target_list:list):
    folder = Path(folder_path)
    # Using tqdm here too for stats generation
    json_files = list(folder.rglob("*.json"))
    for filePath in tqdm(json_files, desc=f"Counting {folder.name}", leave=False):
        try:
            with open(filePath, encoding="utf-8") as file:
                data = json.load(file)
            for token in data.get("tokens", []):
                word = ""
                for target in target_list:
                    if word is not None: token = token.get(target)
                word = token
                if word is not None:
                    if isinstance(word, dict):
                        for k, v in word.items(): wordCounter[k][v] += 1
                    else:
                        wordCounter[word] += 1
        except Exception: pass

# ==========================================
#          STATS & GRAPHS
# ==========================================
# (Condensed graph functions)

def save_tower_graph(title, x_label, y_label, values_list:list[tuple[str, dict]], out_dir="plots", ext="png"):
    all_keys = set().union(*(d.keys() for _, d in values_list))
    
    # 1. Filter out empty keys
    keys = sorted([k for k in all_keys if str(k).strip()], key=str)
    
    if not keys:
        print(f"Skipping empty graph: {title}")
        return

    x_pos = np.arange(len(keys))
    width = 0.8 / max(len(values_list), 1)

    plt.figure(figsize=(12, 6))
    for i, (name, d) in enumerate(values_list):
        plt.bar(x_pos + i * width, [d.get(k, 0) for k in keys], width, label=name)

    centers = x_pos + width*(len(values_list)-1)/2
    
    # 2. SANITIZATION (The Fix)
    # This regex removes: $  \  {  }  ^  _
    clean_labels = [re.sub(r'[$_^{}\\]', '', str(k)) for k in keys]
    
    plt.xticks(centers, clean_labels, rotation=45, ha="right")
    plt.title(title); plt.xlabel(x_label); plt.ylabel(y_label); plt.legend()
    plt.tight_layout()
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{title}.{ext}"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved graph: {title}")

def print_7b_stats():
    # ... (Same logic as before) ...
    pass # Implementation assumed from previous steps

def gather_advanced_stats(json_folder_str: str):
    total_depth, total_unique, file_count, total_tokens = 0, 0, 0, 0
    counts_gi = Counter()
    counts_poss = Counter()

    folder = Path(json_folder_str)
    files = list(folder.rglob("*.json"))
    
    for file_path in tqdm(files, desc=f"Adv Stats {folder.name}", leave=False):
        try:
            with open(file_path, encoding="utf-8") as f:
                tokens = json.load(f).get("tokens", [])
                if not tokens: continue
                
                total_depth += stat_6(tokens)
                total_unique += stat_5(tokens)
                file_count += 1
                total_tokens += len(tokens)
                
                gi = stat_8b(tokens)
                counts_gi["Gerund"] += len(gi["gerund"])
                counts_gi["Infinitive"] += len(gi["infinitive"])
                
                poss = stat_8c(tokens)
                counts_poss["Construct State"] += len(poss["construct_state"])
                counts_poss["Shel Phrase"] += len(poss["prepositional_shel"])
        except Exception: pass

    avgs = {"Tree Depth": total_depth/file_count, "Unique Words": total_unique/file_count} if file_count else {}
    gi_dist = {k: v/total_tokens for k,v in counts_gi.items()} if total_tokens else {}
    poss_dist = {k: v/total_tokens for k,v in counts_poss.items()} if total_tokens else {}
    return avgs, gi_dist, poss_dist

def printStatistics():
    print("Calculating statistics...")
    word_dicts = [Counter() for _ in range(4)]
    length_dicts = [Counter() for _ in range(4)]
    
    avg_stats, gi_stats, poss_stats = [], [], []
    paths = [
        ("Bibal", InputBibalRelativePath, JsonFilesBibalRelativePath), 
        ("Mishna", InputMishnaRelativePath, JsonFilesMishnaRelativePath),
        ("Rambam", InputRambamRelativePath, JsonFilesRambamRelativePath), 
        ("Modern", InputModernRelativePath, JsonFilesModernRelativePath)
    ]
    
    for i, (name, inPath, jsonPath) in enumerate(paths):
        print(f"--- {name} ---")
        countWords(jsonPath, word_dicts[i], ["token"])
        
        # Count sentence lengths
        text_files = list(Path(inPath).rglob("*"))
        for f in tqdm(text_files, desc="Sentence Lengths", leave=False):
            if f.is_file() and not f.name.startswith("."):
                try:
                    txt = readXmlText(f) if f.suffix == ".xml" else f.read_text(encoding="utf-8", errors="ignore")
                    iterate_folder_count_scentences_length(txt, [length_dicts[i]], f)
                except: pass

        avgs, gi, poss = gather_advanced_stats(jsonPath)
        avg_stats.append((name, avgs))
        gi_stats.append((name, gi))
        poss_stats.append((name, poss))

    print("Generating Plots...")
    save_tower_graph("Words Frequency", "Word", "Frequency", 
                    [(p[0], stat_1(word_dicts[i])) for i, p in enumerate(paths)])
    save_tower_graph("Biblical vs Rabbinic Terms", "Word", "Frequency", 
                    [(p[0], stat_2(word_dicts[i])) for i, p in enumerate(paths)])
    save_tower_graph("Subordinating conjunctions", "Word", "Frequency", 
                    [(p[0], stat_3(word_dicts[i])) for i, p in enumerate(paths)])
    save_tower_graph("Sentence Length distribution", "Sentence Length", "Frequency", 
                    [(p[0], stat_4(length_dicts[i])) for i, p in enumerate(paths)])
    save_tower_graph("Syntactic Complexity (Average)", "Metric", "Value", avg_stats)
    save_tower_graph("Gerund vs Infinitive Distribution", "Form", "Frequency", gi_stats)
    save_tower_graph("Possessive Constructions", "Type", "Frequency", poss_stats)
    
    print("Done! Check 'plots' folder.")

# ==========================================
#             MENU & MAIN
# ==========================================

def inpufFile(filePath: str, folderPath: str):
    sourceFile = Path(filePath)
    if sourceFile.exists() and sourceFile.is_file():
        dest = Path(folderPath)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sourceFile, dest / sourceFile.name)
        print(f"File saved to {dest}")
    else: print("File not found.")

def verify_functions():
    print("\n--- VERIFICATION MODE ---")
    load_resources() # Ensure model is loaded
    folders = [("Bibal", InputBibalRelativePath), ("Mishna", InputMishnaRelativePath), 
            ("Rambam", InputRambamRelativePath), ("Modern", InputModernRelativePath)]
    
    for label, folder_path in folders:
        path_obj = Path(folder_path)
        found_file = next((f for f in path_obj.rglob("*") if f.is_file() and not f.name.startswith(".")), None)
        if not found_file: continue

        print(f"[{label}] Found file: {found_file.name}")
        try:
            if found_file.suffix.lower() == ".xml": text = readXmlText(found_file)
            else:
                try: text = found_file.read_text(encoding="utf-8")
                except: text = found_file.read_text(encoding="cp1255")
            
            short_text = sliceText(text)[0] if text else ""
            if not short_text: continue
            
            print(f"   Input: \"{short_text[:40]}...\"")
            json_result = analyzeText(short_text)
            tokens = json_result.get("tokens", [])
            
            print(f"   > Depth: {stat_6(tokens)}")
            print(f"   > Unique: {stat_5(tokens)}")
            print(f"   > Gerunds: {stat_8b(tokens)}")
            print(f"   > Possessives: {stat_8c(tokens)}")
            print("-" * 30)
        except Exception as e: print(f"Error: {e}")

def runAnalysis():
    load_resources() # Load ONCE before loop
    print("Starting Analysis...")
    iterateFolder(InputBibalRelativePath, iterate_folder_slice_text, [JsonFilesBibalRelativePath])
    iterateFolder(InputMishnaRelativePath, iterate_folder_slice_text, [JsonFilesMishnaRelativePath])
    iterateFolder(InputRambamRelativePath, iterate_folder_slice_text, [JsonFilesRambamRelativePath])
    iterateFolder(InputModernRelativePath, iterate_folder_slice_text, [JsonFilesModernRelativePath])            
    print("Analysis Complete.")

def menuRun():
    menu = [("Load File", lambda: inpufFile(input("Path: "), input("Dest: "))),
            ("Run Analysis", runAnalysis),
            ("Run Verification", verify_functions),
            ("Print Statistics", printStatistics),
            ("Exit", None)]
    
    while True:
        print("\n=== MENU ===")
        for i, (cmd, _) in enumerate(menu): print(f"{i} - {cmd}")
        choice = input("Select: ")
        if choice.isdigit():
            idx = int(choice)
            if idx == len(menu) - 1: break
            if 0 <= idx < len(menu): menu[idx][1]()
        else: print("Invalid number.")

def buildDirectoryPath():
    for p in [InputBibalRelativePath, InputMishnaRelativePath, InputRambamRelativePath, InputModernRelativePath,
            JsonFilesBibalRelativePath, JsonFilesMishnaRelativePath, JsonFilesRambamRelativePath, JsonFilesModernRelativePath]:
        Path(p).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    buildDirectoryPath()
    menuRun()