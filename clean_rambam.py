import re
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
TARGET_FOLDER = "InputFiles/hazal/rambam" 

# Define the symbols to clean (remove space before them).
# Current list: Dot (.), Comma (,), Single Quote/Geresh ('), Colon (:), Semicolon (;)
# You can add more inside the brackets [] if needed.
SYMBOLS_PATTERN = r"([.,':;])"

def clean_file(file_path):
    # 1. Read the file (Try UTF-8, then CP1255)
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = file_path.read_text(encoding="cp1255")
        except Exception as e:
            print(f"❌ Could not read {file_path.name}: {e}")
            return False

    if not text.strip():
        return False

    # 2. The Fix: Find space(s) followed by one of our symbols, and keep only the symbol.
    # r'\s+' matches one or more spaces.
    # ([...]) captures the symbol found.
    # r'\1' replaces the whole match with just the captured symbol (removing the space).
    new_text = re.sub(r'\s+' + SYMBOLS_PATTERN, r'\1', text)

    # 3. Check if anything actually changed
    if new_text != text:
        # Write the clean text back to the SAME file (Overwrite)
        file_path.write_text(new_text, encoding="utf-8")
        return True
    
    return False

def main():
    folder = Path(TARGET_FOLDER)
    if not folder.exists():
        print(f"Error: Folder '{TARGET_FOLDER}' not found.")
        return

    print(f"🧹 Cleaning text files in: {folder.absolute()}")
    
    # Find all .txt files recursively
    files = list(folder.rglob("*.txt"))
    modified_count = 0

    for file in tqdm(files, desc="Cleaning files"):
        if clean_file(file):
            modified_count += 1

    print(f"\n✅ Done!")
    print(f"Processed {len(files)} files.")
    print(f"Cleaned and updated {modified_count} files.")
    print("You can now run the main analysis script.")

if __name__ == "__main__":
    main()