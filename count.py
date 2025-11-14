import os
from collections import Counter

DATASET_DIR = "C:/SumV/SignLanguageToText/dataSet"

counts = {}
for cls in os.listdir(DATASET_DIR):
    cls_path = os.path.join(DATASET_DIR, cls)
    if os.path.isdir(cls_path):
        counts[cls] = len(os.listdir(cls_path))

# Sort and display
for k, v in sorted(counts.items(), key=lambda x: x[0]):
    print(f"{k}: {v}")
