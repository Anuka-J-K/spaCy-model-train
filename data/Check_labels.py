import spacy
from spacy.tokens import DocBin

db = DocBin().from_disk("data/train.spacy")
nlp = spacy.blank("en")
docs = list(db.get_docs(nlp.vocab))
labels = set()
for doc in docs[:100]:   # check first 100 docs
    for ent in doc.ents:
        labels.add(ent.label_)
print("Labels in train.spacy:", labels)