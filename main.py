from __future__ import annotations # for future proof typing - easier and handles older python versions
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import List, Dict, Any
import datetime, hashlib, json, re, shutil
import xml.etree.ElementTree as ET

MODEL_ID = "dicta-il/dictabert-tiny-joint"

maxNumOfWords = 12
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


def readXmlText(path: Path) -> str:
    root = ET.parse(path).getroot()
    parts = []
    for text in root.itertext():
        cleaned = text.strip()
        if cleaned:
            parts.append(cleaned)

    return " ".join(parts)


def runAnalysis():
    iterateFolder(InputBibalRelativePath, JsonFilesBibalRelativePath)
    iterateFolder(InputMishnaRelativePath, JsonFilesMishnaRelativePath)
    iterateFolder(InputRambamRelativePath, JsonFilesRambamRelativePath)

    modernFolderPath = Path(InputModernRelativePath)
    modernFolderPathJson = Path(JsonFilesModernRelativePath)
    for file in modernFolderPath.iterdir():
        if file.is_dir():
            filePath = Path(file)
            inputPath = modernFolderPath / filePath.name
            outputPath =modernFolderPathJson / filePath.name
            iterateFolder(inputPath, outputPath)
            


def iterateFolder(inputFolder, outputFolder):
    inputFolderPath = Path(inputFolder)
    for file in inputFolderPath.iterdir():
        if file.is_file():
            if file.suffix.lower() == ".xml":
                sampleText = readXmlText(file)
            else:
                sampleText = file.read_text(encoding="utf-8")
            textInSize = sliceText(sampleText)
            for index, text in enumerate(textInSize):
                result = analyzeText(text)

                save_json(result, outputFolder, str(index) + file.name)


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
