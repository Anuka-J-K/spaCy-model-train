import pandas as pd
import spacy
from spacy.tokens import DocBin
import os

# Load the dataset from Resume.csv
df = pd.read_csv('Resume.csv')

# Create a blank English NLP model (no download needed, instant!)
nlp = spacy.blank("en")

doc_bin = DocBin()

# Process each resume
processed_count = 0
for idx, row in df.iterrows():
    resume_text = row['Resume_str']
    
    if not isinstance(resume_text, str) or len(resume_text.strip()) == 0:
        continue
    
    try:
        # Process text with blank model (fast tokenization only)
        doc = nlp(resume_text)
        
        # Add doc to bin
        doc_bin.add(doc)
        processed_count += 1
        
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} resumes...")
    except Exception as e:
        print(f"Skipping resume {idx}: {e}")
        continue

# Create output directory if needed
os.makedirs("data", exist_ok=True)

doc_bin.to_disk("data/train.spacy")
print(f"\n✓ Conversion complete!")
print(f"✓ Converted {processed_count} resumes to spaCy format")
print(f"✓ Saved to: data/train.spacy")