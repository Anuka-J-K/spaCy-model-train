import spacy
from spacy.tokens import DocBin
from collections import Counter

def check_file(path):
    print(f"--- Checking {path} ---")
    try:
        db = DocBin().from_disk(path)
        nlp = spacy.blank("en")
        docs = list(db.get_docs(nlp.vocab))
        print(f"Number of docs: {len(docs)}")
        
        label_counts = Counter()
        for doc in docs:
            for ent in doc.ents:
                label_counts[ent.label_] += 1
        
        print("Label counts:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
            
        if not label_counts:
            print("  NO ENTITIES FOUND!")
    except Exception as e:
        print(f"  Error reading file: {e}")
    print()

check_file("data/train.spacy")
check_file("data/train_final.spacy")
check_file("data/dev.spacy")
