import json
import os
import string
import spacy
from spacy.tokens import DocBin, Span
from spacy.util import filter_spans

JSON_SOURCE = "Entity Recognition in Resumes.json"
CSV_SOURCE = "Resume.csv"
OUTPUT_PATH = "data/train.spacy"

LANGUAGE_TERMS = {
    "english", "hindi", "french", "german", "spanish", "japanese",
    "chinese", "portuguese", "arabic", "russian", "kannada",
    "tamil", "telugu", "malayalam", "urdu","spanish", "italian", "dutch", "swedish", "norwegian",
    "danish", "finnish", "polish", "czech", "greek", "hungarian",
    "sinhala", "vietnamese", "thai", "indonesian", "bengali", "marathi", "gujarati",
}

LABEL_MAP = {
    'Designation': 'JOB_TITLE',
    'Companies worked at': 'ORG',
    'College Name': 'SCHOOL',
    'Degree': 'DEGREE',
    'Certificate': 'CERTIFICATE',
    'Location': 'GPE',
    'Graduation Year': 'DATE',
    'Languages': 'LANGUAGE',
    'Skills': 'SKILL',
    'Email Address': 'EMAIL',
    'Name': 'NAME',
}

CERTIFICATE_KEYWORDS = ('certificate',)


def load_json_annotations(path: str):
    annotations = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                annotations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return annotations


def find_language_spans(doc):
    spans = []
    for token in doc:
        text = token.text.lower().strip(".,;:-")
        if text in LANGUAGE_TERMS:
            spans.append(Span(doc, token.i, token.i + 1, label="LANGUAGE"))
    return spans


def normalize_label(label, text=''):
    if isinstance(label, (list, tuple)) and label:
        label = str(label[0])
    elif not isinstance(label, str):
        return None

    label = label.strip()
    mapped = LABEL_MAP.get(label)
    if not mapped:
        return None

    if mapped == "DEGREE" and isinstance(text, str) and any(keyword in text.lower() for keyword in CERTIFICATE_KEYWORDS):
        return "CERTIFICATE"

    return mapped


def build_docbin_from_json(nlp, items):
    doc_bin = DocBin()
    label_set = set()
    processed_count = 0

    for item in items:
        text = item.get("content") or item.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            continue

        doc = nlp(text)
        ents = []

        for ann in item.get("annotation", []):
            for point in ann.get("points", []):
                start = point.get("start")
                end = point.get("end") + 1  # JSON end is inclusive, spaCy end is exclusive
                if start is None or end is None:
                    continue

                point_text = point.get("text", "")
                label = normalize_label(ann.get("label"), point_text)
                if not label:
                    continue

                # Trim whitespace and punctuation from the span edges
                strip_chars = string.whitespace + string.punctuation
                while start < end and text[start] in strip_chars:
                    start += 1
                while end > start and text[end-1] in strip_chars:
                    end -= 1

                if start >= end:
                    continue

                # Initial span creation
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
                
                if span:
                    # Token-level trimming: remove leading/trailing whitespace or punctuation tokens
                    # This is the most robust way to satisfy spaCy's training requirements
                    while len(span) > 0 and (span[0].is_space or span[0].is_punct):
                        span = span[1:]
                    while len(span) > 0 and (span[-1].is_space or span[-1].is_punct):
                        span = span[:-1]

                if not span or len(span) == 0:
                    continue

                ents.append(span)
                label_set.add(label)

        if not any(span.label_ == "LANGUAGE" for span in ents):
            language_spans = find_language_spans(doc)
            if language_spans:
                ents.extend(language_spans)
                label_set.add("LANGUAGE")

        if ents:
            ents = filter_spans(ents)
            doc.ents = ents

        doc_bin.add(doc)
        processed_count += 1

        if processed_count % 100 == 0:
            print(f"Processed {processed_count} annotated resumes...")

    return doc_bin, processed_count, label_set


def build_docbin_from_csv(nlp, path: str):
    import csv

    doc_bin = DocBin()
    processed_count = 0

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            resume_text = row.get("Resume_str")
            if not isinstance(resume_text, str) or not resume_text.strip():
                continue

            doc = nlp(resume_text)
            language_spans = find_language_spans(doc)
            if language_spans:
                doc.ents = filter_spans(language_spans)
            doc_bin.add(doc)
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} resumes...")

    return doc_bin, processed_count


def main():
    nlp = spacy.blank("en")

    if os.path.exists(JSON_SOURCE):
        print(f"Loading annotations from {JSON_SOURCE}")
        items = load_json_annotations(JSON_SOURCE)
        doc_bin, processed_count, label_set = build_docbin_from_json(nlp, items)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        doc_bin.to_disk(OUTPUT_PATH)

        print("\nConversion complete!")
        print(f"Converted {processed_count} annotated resumes to spaCy format")
        print(f"Saved to: {OUTPUT_PATH}")
        print(f"Found entity labels: {sorted(label_set)}")

    elif os.path.exists(CSV_SOURCE):
        print(f"No annotation JSON found. Loading raw text from {CSV_SOURCE}")
        doc_bin, processed_count = build_docbin_from_csv(nlp, CSV_SOURCE)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        doc_bin.to_disk(OUTPUT_PATH)

        print("\nConversion complete!")
        print(f"Converted {processed_count} resumes to spaCy format")
        print(f"Saved to: {OUTPUT_PATH}")
        print("No entity labels were added because no JSON annotations were available.")

    else:
        print(f"ERROR: Neither {JSON_SOURCE} nor {CSV_SOURCE} was found.")


if __name__ == "__main__":
    main()
