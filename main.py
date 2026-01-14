from __future__ import annotations # for future proof typing - easier and handles older python versions
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import List, Dict, Any
import json, re, shutil
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

MODEL_ID = "dicta-il/dictabert-tiny-joint"

maxNumOfWords = 200
_tokenizer = None
_model = None
menu = [
    ("Load File", lambda: inpufFile(input("Enter the Path to the file : "),
                                    input("Enter the Path to destination: "))),
    ("Run Analysis", lambda: runAnalysis()),
    ("Print Statistics", lambda: printStatistics()),
    ("Exit", None) 
]
wordsFor_stat_2 = []
wordsFor_stat_3 = ["ש", "אשר", "כי", "כאשר", "מאשר", "אם", "פן"]

InputBibalRelativePath = "InputFiles/bibal"
InputMishnaRelativePath = "InputFiles/hazal/mishna"
InputRambamRelativePath = "InputFiles/hazal/rambam"
InputModernRelativePath = "InputFiles/modern"

JsonFilesBibalRelativePath = "JsonFiles/bibal"
JsonFilesMishnaRelativePath = "JsonFiles/hazal/mishna"
JsonFilesRambamRelativePath = "JsonFiles/hazal/rambam"
JsonFilesModernRelativePath = "JsonFiles/modern"


def readXmlText(path: Path) -> str:
    root = ET.parse(path).getroot()
    parts = []
    for text in root.itertext():
        cleaned = text.strip()
        if cleaned:
            parts.append(cleaned)

    return " ".join(parts)


def runAnalysis():
    iterateFolder(InputBibalRelativePath, iterate_folder_slice_text, [JsonFilesBibalRelativePath])
    iterateFolder(InputMishnaRelativePath, iterate_folder_slice_text, [JsonFilesMishnaRelativePath])
    iterateFolder(InputRambamRelativePath, iterate_folder_slice_text, [JsonFilesRambamRelativePath])
    iterateFolder(InputModernRelativePath, iterate_folder_slice_text, [JsonFilesModernRelativePath])            

def iterateFolder(inputFolder, func, func_vars: list):
    inputFolderPath = Path(inputFolder)
    for file in inputFolderPath.iterdir():
        if file.is_file():
            if file.suffix.lower() == ".xml":
                sampleText = readXmlText(file)
            else:
                sampleText = file.read_text(encoding="utf-8")
            func(sampleText, func_vars, file)

def iterate_folder_slice_text(sampleText, func_vars, file):
    outputFolder = func_vars[0]
    textInSize = sliceText(sampleText)
    for index, text in enumerate(textInSize):
        result = analyzeText(text)
        save_json(result, outputFolder, str(index) + file.name)

def iterate_folder_count_scentences_length(sampleText, func_vars, file):
    output_dict = func_vars[0]
    splitSectences = re.split(r'([.!?])\s*', sampleText)
    sentences = [splitSectences[2*i].strip() + splitSectences[2*i+1] for i in range(0, len(splitSectences)//2)]
    if len(splitSectences) % 2 != 0:
        sentences.append(splitSectences[-1]) 
    for sentence in sentences:
        output_dict[len(sentence.split(" "))] += 1


def sliceText(text: str) -> List[str]:
    splitSectences = re.split(r'([.!?])\s*', text)
    sentences = [splitSectences[2*i].strip() + splitSectences[2*i+1] for i in range(0, len(splitSectences)//2)]
    if len(splitSectences) % 2 != 0:
        sentences.append(splitSectences[-1]) 
    output = []
    currChunck = ""
    currChunckSize = 0
    for sentence in sentences:
        words = sentence.split(" ")
        if currChunckSize + len(words) <= maxNumOfWords :
            currChunck += " " + sentence
            currChunckSize += len(words)
        else:
            output.append(currChunck)
            currChunck = sentence
            currChunckSize = len(words)
    
    if currChunck != "":
        output.append(currChunck)

    return output


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,   # IMPORTANT: this model uses custom code on the Hub
            # Enable/disable heads as needed (saves time if you only need some)
            do_lex=False,
            do_syntax=True,
            do_ner=True,
            do_prefix=False,
            do_morph=True,
        )
        _model.eval()  # switch to evaluate mode and not training the model itself.
    return _tokenizer, _model


def analyzeText(text: str, output_style: str = "json") -> Dict[str, Any]:
    tokenizer, model = _load_model()
    result = model.predict([text], tokenizer, output_style=output_style)
    # model.predict returns a list (one per input string)
    if isinstance(result, list):
        return result[0] 
    return result

def save_json(obj: Any, base_dir: str, filename: str):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    jsonName = filename.split(".")[0] + ".json"
    path = base / jsonName
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)
        # ^ ensure_ascii=False keeps Hebrew readable; indent=2 is nicer for diagnostics


def inpufFile(filePath: str, folderPath: str):
    sourceFile = Path(filePath)
    if not sourceFile.exists():
        print(f"No file found with name {filePath}")
    else:
        if not sourceFile.is_file():
            print(f"Type Error - {filePath}  is not a file.")

        fileDestonationFolder = Path(folderPath)
        fileDestonationFolder.mkdir(parents=True, exist_ok=True)
        outputFile = fileDestonationFolder / sourceFile.name
        shutil.copy2(sourceFile, outputFile)
        
        if not outputFile.exists():
            print(f"Failed to save the file to {folderPath}. Please try again.")

def countWords(folder_path:str, wordCounter, target_list:list) -> dict[str, int]:
    """
    target list must contain the path of inside each tocken to the desired data, ending in the key to the data in the token
    """
    folder = Path(folder_path)
    for filePath in folder.glob("*.json"):
        with open(filePath, encoding="utf-8") as file:
            data = json.load(file)
        
        for token in data.get("tokens", []):
            word = ""
            for target in target_list:
                if word is not None: 
                    token = token.get(target)
            word = token
            if word is not None:
                if isinstance(word, dict):
                    for key, value in word.items():
                        wordCounter[key][value] += 1
                else:
                    wordCounter[word] += 1

def stat_1(wordDict: dict[str, int]) -> list:
    num_of_words = sum(wordDict.values())
    sub_dict = {k: float(wordDict[k])/num_of_words for k in wordDict}
    return sub_dict

def stat_2(wordDict: dict[str, int]) -> dict:
    num_of_words = sum(wordDict.values())
    sub_dict = {k: float(wordDict[k])/num_of_words for k in wordsFor_stat_2 if k in wordDict}
    return sub_dict

def stat_3(wordDict: dict[str, int]) -> dict:
    num_of_words = sum(wordDict.values())
    sub_dict = {k: float(wordDict[k])/num_of_words for k in wordsFor_stat_3 if k in wordDict}
    return sub_dict

def stat_4(lengthDict: dict[int, int]) -> dict:
    num_of_scentences = sum(lengthDict.values())
    output_dict = {k: float(lengthDict[k])/num_of_scentences for k in lengthDict}
    return output_dict

def stat_7_8a(dir_path: str, path_to_objective)->dict:
    word_dict = Counter()
    countWords(dir_path, word_dict, path_to_objective)
    num_of_words = sum(word_dict.values())
    sub_dict = {k: float(word_dict[k])/num_of_words for k in word_dict}
    return sub_dict

def stat_7b(dir_path: str)->dict:
    word_dict: dict[str, Counter] = {}
    word_dict["Gender"] = Counter()
    word_dict["Number"] = Counter()
    word_dict["Person"] = Counter()
    word_dict["Tense"] = Counter()
    countWords(dir_path, word_dict, ["morph", "feats"])

    gender_dict = dict(word_dict["Gender"])
    num_dict = dict(word_dict["Number"])
    person_dict = dict(word_dict["Person"])
    tense_dict = dict(word_dict["Tense"])

    gender_num_of_words = sum(gender_dict.values())
    numbers_num_of_words = sum(num_dict.values())
    person_num_of_words = sum(person_dict.values())
    tense_num_of_words = sum(tense_dict.values())

    gender_sub_dict = {k: float(gender_dict[k])/gender_num_of_words for k in gender_dict}
    numbers_sub_dict = {k: float(num_dict[k])/numbers_num_of_words for k in num_dict}
    person_sub_dict = {k: float(person_dict[k])/person_num_of_words for k in person_dict}
    tense_sub_dict = {k: float(tense_dict[k])/tense_num_of_words for k in tense_dict}
    return [gender_sub_dict, numbers_sub_dict, person_sub_dict, tense_sub_dict]
    
def print_7b_stats():
    bibal_gender_sub_dict, bibal_numbers_sub_dict, bibal_person_sub_dict, bibal_tense_sub_dict = stat_7b(JsonFilesBibalRelativePath)
    mishna_gender_sub_dict, mishna_numbers_sub_dict, mishna_person_sub_dict, mishna_tense_sub_dict = stat_7b(JsonFilesMishnaRelativePath)
    rambam_gender_sub_dict, rambam_numbers_sub_dict, rambam_person_sub_dict, rambam_tense_sub_dict = stat_7b(JsonFilesRambamRelativePath)
    modern_gender_sub_dict, modern_numbers_sub_dict, modern_person_sub_dict, modern_tense_sub_dict = stat_7b(JsonFilesModernRelativePath)
    save_tower_graph("Gender Feature distribution", "Gender Feature", "Frequency", 
                    [("Bibal" ,bibal_gender_sub_dict), ("Mishna" ,mishna_gender_sub_dict), 
                    ("Rambam" ,rambam_gender_sub_dict), ("Modern" ,modern_gender_sub_dict)])
    save_tower_graph("Number Feature distribution", "Number Feature", "Frequency", 
                    [("Bibal" ,bibal_numbers_sub_dict), ("Mishna" ,mishna_numbers_sub_dict), 
                    ("Rambam" ,rambam_numbers_sub_dict), ("Modern" ,modern_numbers_sub_dict)])
    save_tower_graph("Person Feature distribution", "Person Feature", "Frequency", 
                    [("Bibal" ,bibal_person_sub_dict), ("Mishna" ,mishna_person_sub_dict), 
                    ("Rambam" ,rambam_person_sub_dict), ("Modern" ,modern_person_sub_dict)])
    save_tower_graph("Tense Feature distribution", "Tense Feature", "Frequency", 
                    [("Bibal" ,bibal_tense_sub_dict), ("Mishna" ,mishna_tense_sub_dict), 
                    ("Rambam" ,rambam_tense_sub_dict), ("Modern" ,modern_tense_sub_dict)])

def save_tower_graph(title, x_label, y_label, values_list:list[tuple[str, dict]], out_dir="plots", ext="png"):
    all_keys = set()
    for _, d in values_list:
        all_keys.update(d.keys())
    keys = sorted(all_keys, key=str)

    x_pos = np.arange(len(keys))
    n_series = len(values_list)
    width = 0.8 / max(n_series, 1)

    plt.figure(figsize=(12, 6))

    for i, (series_name, d) in enumerate(values_list):
        y = [d.get(k, 0) for k in keys]  # missing -> 0
        plt.bar(x_pos + i * width, y, width, label=series_name)

    centers = x_pos + width * (n_series - 1) / 2
    plt.xticks(centers, [str(k) for k in keys], rotation=45, ha="right")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{title}.{ext}"

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return str(out_path)


def printStatistics():
    word_dicts = [Counter(), Counter(), Counter(), Counter()]
    length_dicts = [Counter(), Counter(), Counter(), Counter()]
    paths = [(InputBibalRelativePath, JsonFilesBibalRelativePath), (InputMishnaRelativePath, JsonFilesMishnaRelativePath),
             (InputRambamRelativePath, JsonFilesRambamRelativePath), (InputModernRelativePath, JsonFilesModernRelativePath)]
    for i, (inPath, jsonPath) in enumerate(paths):    
        countWords(jsonPath, word_dicts[i], ["token"])
        iterateFolder(inPath, iterate_folder_count_scentences_length, [length_dicts[i]])
    
    save_tower_graph("Words Frequency", "Word", "Frequency", [("Bibal" ,stat_1(word_dicts[0])), ("Mishna" ,stat_1(word_dicts[1])), ("Rambam" ,stat_1(word_dicts[2])), ("Modern" ,stat_1(word_dicts[3]))])
    save_tower_graph("Edith Doron Special Words", "Word", "Frequency", [("Bibal" ,stat_2(word_dicts[0])), ("Mishna" ,stat_2(word_dicts[1])), ("Rambam" ,stat_2(word_dicts[2])), ("Modern" ,stat_2(word_dicts[3]))])
    save_tower_graph("Subordinating conjunctions", "Word", "Frequency", [("Bibal" ,stat_3(word_dicts[0])), ("Mishna" ,stat_3(word_dicts[1])), ("Rambam" ,stat_3(word_dicts[2])), ("Modern" ,stat_3(word_dicts[3]))])
    save_tower_graph("Scentences Length distribution", "Scentence Length", "Frequency", [("Bibal" ,stat_4(length_dicts[0])), ("Mishna" ,stat_4(length_dicts[1])), ("Rambam" ,stat_4(length_dicts[2])), ("Modern" ,stat_4(length_dicts[3]))])
    save_tower_graph("Words Category distribution", "Word category", "Frequency", [("Bibal" ,stat_4(length_dicts[0])), ("Mishna" ,stat_4(length_dicts[1])), ("Rambam" ,stat_4(length_dicts[2])), ("Modern" ,stat_4(length_dicts[3]))])
    save_tower_graph("Words Category distribution", "Word category", "Frequency", 
                    [("Bibal" ,stat_7_8a(JsonFilesBibalRelativePath, ["morph", "pos"])), 
                    ("Mishna" ,stat_7_8a(JsonFilesMishnaRelativePath, ["morph", "pos"])), 
                    ("Rambam" ,stat_7_8a(JsonFilesRambamRelativePath, ["morph", "pos"])), 
                    ("Modern" ,stat_7_8a(JsonFilesModernRelativePath, ["morph", "pos"]))])
    save_tower_graph("Types of syntactic relations distribution", "Type of syntactic relations", "Frequency", 
                    [("Bibal" ,stat_7_8a(JsonFilesBibalRelativePath, ["syntax", "dep_func"])), 
                    ("Mishna" ,stat_7_8a(JsonFilesMishnaRelativePath, ["syntax", "dep_func"])), 
                    ("Rambam" ,stat_7_8a(JsonFilesRambamRelativePath, ["syntax", "dep_func"])), 
                    ("Modern" ,stat_7_8a(JsonFilesModernRelativePath, ["syntax", "dep_func"]))])
    print_7b_stats()

def menuRun():
    finish = False
    while not finish:
        for i, (command, func) in enumerate(menu):
            print(f"{i} - {command}")
        choice = input("Please enter the number of command to be executed: ")

        if not choice.isdigit():
            print("Please enter a number ")
        else:
            choiceIndex = int(choice)
            if choiceIndex >= len(menu):
                print("Please enter a number corresponding to the comands ")
            elif choiceIndex == (len(menu)-1):
                finish = True
            else:
                func = menu[choiceIndex][1]
                if func is not None:
                    func()

def buildDirectoryPath():
    inputBibalPath = Path(InputBibalRelativePath)
    inputBibalPath.mkdir(parents=True, exist_ok=True)
    inputMishnalPath = Path(InputMishnaRelativePath)
    inputMishnalPath.mkdir(parents=True, exist_ok=True)
    inputRambamPath = Path(InputRambamRelativePath)
    inputRambamPath.mkdir(parents=True, exist_ok=True)
    inputmodernPath = Path(InputModernRelativePath)
    inputmodernPath.mkdir(parents=True, exist_ok=True)

    jsonFilesBibalPath = Path(JsonFilesBibalRelativePath)
    jsonFilesBibalPath.mkdir(parents=True, exist_ok=True)
    jsonFilesMishnalPath = Path(JsonFilesMishnaRelativePath)
    jsonFilesMishnalPath.mkdir(parents=True, exist_ok=True)
    jsonFilesRambamPath = Path(JsonFilesRambamRelativePath)
    jsonFilesRambamPath.mkdir(parents=True, exist_ok=True)
    jsonFilesmodernPath = Path(JsonFilesModernRelativePath)
    jsonFilesmodernPath.mkdir(parents=True, exist_ok=True)

def main():
    buildDirectoryPath()
    menuRun()

if __name__ == "__main__":
    main()
