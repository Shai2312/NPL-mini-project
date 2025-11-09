
from transformers import AutoTokenizer, AutoModel
import json

MODEL_ID = "dicta-il/dictabert-tiny-joint"  # or a local path if downloaded with huggingface-cli

def main():
    # Load tokenizer and model (no debugging of transformers; just use it)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,   # required: uses custom model code on the Hub
        # Choose which heads to run (set True/False as needed):
        do_lex=False,
        do_syntax=True,
        do_ner=True,
        do_prefix=False,
        do_morph=True,
    )

    # Example Hebrew texts
    texts = [
        "אפרים קישון פרסם מאמרים הומוריסטיים בשנת 1948.",
        "אירוע נדיר התקיים בתל אביב בחוף הים.",
    ]

    # Use the model's convenience method exposed via remote code
    # output_style can be "json", "ud", or "iahlt_ud" according to the model card.
    result = model.predict(texts, tokenizer, output_style="json")

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
