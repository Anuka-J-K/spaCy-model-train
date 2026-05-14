import json
import os
import string
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

# Constants
SOURCE_FILE = "train.json"
OUTPUT_PATH = "data/train_v2.spacy"

LABEL_MAP = {
    'DESIGNATION': 'JOB_TITLE',
    'COMPANY': 'ORG',
    'LOCATION': 'GPE',
    'SKILL': 'SKILL',
    'EDUCATION': 'DEGREE',
    'PERSON': 'NAME',
    'EMAIL': 'EMAIL',
    'LANGUAGE': 'LANGUAGE',
    'CERTIFICATION': 'CERTIFICATE',
}

def convert():
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: {SOURCE_FILE} not found.")
        return

    print(f"Loading {SOURCE_FILE}...")
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Converting {len(data)} records...")
    processed_count = 0
    fail_count = 0
    label_set = set()

    strip_chars = string.whitespace + string.punctuation

    for item in data:
        text = item.get("text", "")
        if not text:
            continue
        
        # Clean text of surrogates that crash the tokenizer
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        
        doc = nlp(text)
        ents = []
        
        for start, end, raw_label in item.get("annotations", []):
            label = LABEL_MAP.get(raw_label)
            if not label:
                continue
            
            # Sanitization (Double Check)
            while start < end and text[start] in strip_chars:
                start += 1
            while end > start and text[end-1] in strip_chars:
                end -= 1
            
            if start >= end:
                continue
                
            # Create span
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if not span:
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
            
            if span:
                # Token-level trimming
                while len(span) > 0 and (span[0].is_space or span[0].is_punct):
                    span = span[1:]
                while len(span) > 0 and (span[-1].is_space or span[-1].is_punct):
                    span = span[:-1]
                
                if span and len(span) > 0:
                    ents.append(span)
                    label_set.add(label)
            else:
                fail_count += 1

        if ents:
            doc.ents = filter_spans(ents)
        
        doc_bin.add(doc)
        processed_count += 1
        
        if processed_count % 500 == 0:
            print(f"Processed {processed_count}...")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    doc_bin.to_disk(OUTPUT_PATH)
    
    print("\nConversion complete!")
    print(f"Successfully converted {processed_count} resumes.")
    print(f"Skipped {fail_count} invalid spans.")
    print(f"Labels found: {sorted(list(label_set))}")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    convert()
