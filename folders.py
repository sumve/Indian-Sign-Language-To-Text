import os
import string

# Base directory for dataset
base_dir = "DataSet"

# Define labels (A–Z + 0–9)
labels = list(string.ascii_uppercase) + [str(i) for i in range(10)]

# Create main folder if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

# Create subfolders
for label in labels:
    path = os.path.join(base_dir, label)
    os.makedirs(path, exist_ok=True)

print(f"Folders created successfully inside '{base_dir}'")
