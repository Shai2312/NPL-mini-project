from __future__ import annotations # for future proof typing - easier and handles older python versions
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import List, Dict, Any
import datetime, hashlib, json, re, shutil

MODEL_ID = "dicta-il/dictabert-tiny-joint"

maxNumOfWords = 5
_tokenizer = None
_model = None
menu = [
    ("Load file", lambda: inpufFile(input("Enter the Path to the file : "),
                                    input("Enter the Path to destination: "))),
    ("run analysis", lambda: runAnalysis()),
    ("Exit", None) 
]
InputBibalRelativePath = "InputFiles/bibal"
InputMishnaRelativePath = "InputFiles/hazal/mishna"
InputRambamRelativePath = "InputFiles/hazal/rambam"
InputModernRelativePath = "InputFiles/modern"

JsonFilesBibalRelativePath = "JsonFiles/bibal"
JsonFilesMishnaRelativePath = "JsonFiles/hazal/mishna"
JsonFilesRambamRelativePath = "JsonFiles/hazal/rambam"
JsonFilesModernRelativePath = "JsonFiles/modern"

def runAnalysis():
    """
    Run the analysis for all files in the InputFiles and save the output to the Json files
    """
    sample_text = "ד."
    result = analyzeText(sample_text)
    out_path = save_json(result, base_dir="JsonFiles/Test")
    print(f"Saved JSON to: {out_path}")

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
    """
    loads the tokenizer and model. Does it once at the first use. the loading code itself was given to us
    """
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
    """
    Runs DictaBERT-tiny-joint on a single Hebrew text
    Return:
        a JSON-able dict.
    """
    tokenizer, model = _load_model()
    result = model.predict([text], tokenizer, output_style=output_style)
    # model.predict returns a list (one per input string)
    if isinstance(result, list):
        return result[0] 
    return result

def save_json(obj: Any, base_dir: str = "JsonFiles/Test", filename: str | None = None) -> str:
    """
    Saves JSON (UTF-8, pretty).
    Return:
        the filepath.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)   # auto-create folders, ignores if already exist

    if filename is None:
        # Unique, readable filenames (timestamp + short content hash)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        h  = hashlib.sha1(json.dumps(obj, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
        filename = f"analysis-{ts}-{h}.json"

    path = base / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        # ^ ensure_ascii=False keeps Hebrew readable; indent=2 is nicer for diagnostics
    return str(path)


def inpufFile(filePath: str, folderPath: str) -> int:
    """
    Reads a new file abd adds it to the corresponding folder according to its type.
    return:
        1 if succeeded
        2 if filePath does not exist
        3 if filePath is not a file
        4 if the copied file does not exist
    """
    sourceFile = Path(filePath)
    if not sourceFile.exists():
        return 2
    if not sourceFile.is_file():
        return 3

    fileDestonationFolder = Path(folderPath)
    fileDestonationFolder.mkdir(parents=True, exist_ok=True)
    outputFile = fileDestonationFolder / sourceFile.name
    shutil.copy2(sourceFile, outputFile)
    
    if not outputFile.exists():
        return 4
    return 1

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
    # menuRun()
    buildDirectoryPath()
    print(sliceText("hello! this. is? a! test. for find. out if. it workd. shopud be max. 5 char chunks."))

if __name__ == "__main__":
    main()
