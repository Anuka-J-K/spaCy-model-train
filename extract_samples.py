import csv
import json

with open('Resume.csv', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    samples = []
    for i, row in enumerate(reader):
        if i >= 10: break # Skip first 10
        if i < 7: continue # Take 7, 8, 9
        samples.append(row['Resume_str'])

with open('samples.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f)
