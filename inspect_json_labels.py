import json
from collections import Counter

def inspect_json(path):
    label_counts = Counter()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                item = json.loads(line)
                for ann in item.get("annotation", []):
                    labels = ann.get("label")
                    if isinstance(labels, list):
                        for l in labels:
                            label_counts[l] += 1
                    else:
                        label_counts[labels] += 1
            except:
                continue
    print("Labels in JSON:")
    for l, c in label_counts.items():
        print(f"  {l}: {c}")

inspect_json("Entity Recognition in Resumes.json")
