import json
from collections import Counter

def inspect_new_json(path):
    label_counts = Counter()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Total records: {len(data)}")
            for item in data:
                for ann in item.get("annotations", []):
                    label_counts[ann[2]] += 1
    except Exception as e:
        print(f"Error: {e}")
        
    print("Labels in NEW JSON:")
    for l, c in label_counts.items():
        print(f"  {l}: {c}")

inspect_new_json("train.json")
