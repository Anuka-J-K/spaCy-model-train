import random
import spacy
from spacy.tokens import DocBin

db = DocBin().from_disk("data/train_v2.spacy")
nlp = spacy.blank("en")
docs = list(db.get_docs(nlp.vocab))
random.shuffle(docs)

split = int(len(docs) * 0.8)
train_docs = docs[:split]
dev_docs = docs[split:]

DocBin(docs=train_docs).to_disk("data/train_final.spacy")
DocBin(docs=dev_docs).to_disk("data/dev.spacy")
print(f"Train: {len(train_docs)} docs, Dev: {len(dev_docs)} docs")