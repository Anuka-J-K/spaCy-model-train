import random
import spacy
from spacy.tokens import DocBin

db = DocBin().from_disk("data/train_v2.spacy")
nlp = spacy.blank("en")
docs = list(db.get_docs(nlp.vocab))
random.shuffle(docs)

DocBin(docs=docs[:500]).to_disk("data/train_subset.spacy")
DocBin(docs=docs[500:600]).to_disk("data/dev_subset.spacy")
print("Subset created.")
