import spacy
from spacy.tokens import DocBin

def find_invalid_spans(path):
    print(f"--- Checking {path} ---")
    db = DocBin().from_disk(path)
    nlp = spacy.blank("en")
    docs = list(db.get_docs(nlp.vocab))
    
    invalid_count = 0
    for i, doc in enumerate(docs):
        for ent in doc.ents:
            if ent.text[0].isspace() or ent.text[-1].isspace():
                print(f"Doc {i}, Ent '{ent.text}', Label {ent.label_}")
                invalid_count += 1
    
    print(f"Total invalid spans found: {invalid_count}")

find_invalid_spans("data/train_final.spacy")
