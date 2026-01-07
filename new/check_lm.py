import numpy as np
import os

root = "C:\\SumV\\SignLanguageToText\\DataSet\\Landmarks"

for cls in os.listdir(root):
    cls_path = os.path.join(root, cls)
    if not os.path.isdir(cls_path):
        continue

    files = [f for f in os.listdir(cls_path) if f.endswith(".npy")]
    if not files:
        print(f"[EMPTY] {cls}")
        continue

    arr = np.load(os.path.join(cls_path, files[0]))
    print(cls, arr.shape)
