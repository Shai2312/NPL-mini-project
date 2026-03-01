
# Modern vs. Historical Hebrew: An NLP Analysis

## Overview
This repository contains a comprehensive Natural Language Processing (NLP) research project investigating the historical linguistic roots of Modern Israeli Hebrew. 

The primary research question is the hypothesis proposed by Prof. Edith Doron, which suggests that Modern Hebrew is closer in structure to Biblical Hebrew. Through deep computational analysis of morphology, syntax, and lexicon, this project tests whether Modern Hebrew is effectively a continuation of Biblical Hebrew, or if its deep syntactic "skeleton" actually stems from Rabbinic/Mishnaic Hebrew.

## Architecture & Features
The codebase is built in Python and relies heavily on **DictaBERT** (`dicta-il/dictabert-tiny-joint`) for high-accuracy morphological tagging and dependency parsing of Hebrew text.

### 1. Linguistic Feature Extraction
The pipeline breaks down Hebrew sentences into multiple structural "lenses":
* **Lexical Signatures:** Tracking top-50 word frequencies and specific historical markers.
* **Morphology & POS:** Analyzing the distribution of Parts of Speech, gender, number, person, and tense.
* **Syntax & Dependency:** Calculating Syntactic Tree Depth, Clausal Word Order (V1 vs. V2), and Dependency Relations.
* **Historical Markers:** Infinitive vs. Gerund usage, and Possessive Constructions (Construct-State/Smixut vs. Prepositional/Shel).

### 2. Statistical Distance Metrics
To mathematically prove linguistic proximity, the project calculates multiple similarity/distance metrics across all corpora:
* Manhattan & Euclidean Distances
* Cosine Similarity
* Jaccard & Dice Similarities
* Jensen-Shannon (JS) Divergence 

### 3. Machine Learning Classifier
A robust `LogisticRegression` pipeline (evaluated using 10-Fold Cross Validation) trained to classify text as either Biblical (0) or Rabbinic (1). It tests Modern Hebrew genres (news, blogs, medical, literary) across 5 independent models:
1. **Lexical:** Raw vocabulary.
2. **Morph:** Inflection patterns.
3. **Syntax:** Dependency edges and distances.
4. **Structure:** Abstract POS sequencing.
5. **Linguistic:** Custom mathematical heuristics (Tree depth, unique word ratios, V1/V2 scores).

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/NPL-mini-project.git](https://github.com/your-username/NPL-mini-project.git)
   cd NPL-mini-project

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the example
python main.py
#Menu Options:
    # 1.Load File: Copy text/XML files into the target directories.
    # 2.Run Analysis: Process raw texts through DictaBERT and generate base JSON tokens.
    # 3.Print Statistics: Extract linguistic features, plot Matplotlib bar charts, generate the comprehensive Similarity Distance CSV, and run the ML Classifier on Modern Hebrew datasets.

# 4) Run the classifier
python classifier.py
#Menu Option:
    # 1.Run modern analysis- Automatically processes all Modern Hebrew datasets through the trained Classifier. It outputs a table showing the percentage of "Rabbinic" vs. "Biblical" alignment for each genre across all 5 linguistic lenses.
    # 2.Manual analysis- Type or paste any Hebrew sentence directly into the terminal. The system parses the sentence via DictaBERT in real-time, and all 5 trained models independently classify it as either "Biblical" or "Rabbinic," providing a statistical confidence score for each lens.
```



