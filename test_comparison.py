import spacy
import csv
import os

# Paths
OLD_MODEL = "models/resume_ner/model-best"
NEW_MODEL = "models/resume_ner_v2/model-best"
TEST_CSV = "Resume.csv"

def test_comparison():
    print("Loading models...")
    try:
        nlp_old = spacy.load(OLD_MODEL)
        print("Old model loaded.")
    except:
        print("Old model not found.")
        nlp_old = None
        
    try:
        nlp_new = spacy.load(NEW_MODEL)
        print("New model loaded.")
    except:
        print("New model not found.")
        nlp_new = None

    if not nlp_new:
        return

    # Read a few samples from CSV
    samples = []
    with open(TEST_CSV, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < 5: continue # Skip first 5
            samples.append(row['Resume_str'][:1500]) # Take first 1500 chars
            if len(samples) >= 2: break

    for i, text in enumerate(samples):
        print(f"\n--- Testing Resume Sample {i+1} ---")
        print(f"Text Preview: {text[:100]}...")
        
        if nlp_old:
            doc_old = nlp_old(text)
            print(f"\n[OLD MODEL] Found {len(doc_old.ents)} entities:")
            for ent in doc_old.ents:
                print(f"  - {ent.text} ({ent.label_})")
        
        doc_new = nlp_new(text)
        print(f"\n[NEW MODEL] Found {len(doc_new.ents)} entities:")
        for ent in doc_new.ents:
            print(f"  - {ent.text} ({ent.label_})")

if __name__ == "__main__":
    test_comparison()
