# ============================================
# Landmark-only ISL Training Script
# ============================================

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ============================================
# Config
# ============================================

LANDMARK_DIR = "C:\\SumV\\SignLanguageToText\\DataSet\\Landmarks"     # root landmark dataset folder
EPOCHS = 40
BATCH_SIZE = 128
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================
# Landmark normalization
# ============================================

def normalize_hand(hand):
    """
    Normalize a single hand.
    hand: (21, 3)
    """
    wrist = hand[0]
    hand = hand - wrist

    # scale by palm length (wrist -> middle MCP)
    palm_len = np.linalg.norm(hand[9]) + 1e-6
    hand = hand / palm_len

    return hand

def preprocess_landmark(arr):
    """
    Output is ALWAYS a 126-D vector (2 hands).
    """

    # Ensure shape is (N,21,3)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  # (1,21,3)

    if arr.ndim != 3 or arr.shape[1:] != (21, 3):
        raise ValueError(f"Unexpected landmark shape: {arr.shape}")

    # Keep at most 2 hands
    if arr.shape[0] > 2:
        arr = arr[:2]

    features = []

    # First hand
    h1 = normalize_hand(arr[0])
    features.append(h1.reshape(-1))  # 63

    # Second hand (or zero pad)
    if arr.shape[0] == 2:
        h2 = normalize_hand(arr[1])
        features.append(h2.reshape(-1))  # 63
    else:
        features.append(np.zeros(63, dtype=np.float32))

    return np.concatenate(features)  # (126,)

# ============================================
# Load dataset
# ============================================

X = []
y = []

labels = sorted([
    d for d in os.listdir(LANDMARK_DIR)
    if os.path.isdir(os.path.join(LANDMARK_DIR, d))
])

label_to_idx = {l: i for i, l in enumerate(labels)}

print("Detected classes:", labels)
print("Number of classes:", len(labels))

for label in labels:
    class_dir = os.path.join(LANDMARK_DIR, label)
    for file in os.listdir(class_dir):
        if not file.endswith(".npy"):
            continue

        path = os.path.join(class_dir, file)
        arr = np.load(path)

        feat = preprocess_landmark(arr)
        X.append(feat)
        y.append(label_to_idx[label])

X = np.array(X, dtype=np.float32)
y = to_categorical(y, num_classes=len(labels))

print("Dataset shape:", X.shape, y.shape)

# ============================================
# Train / Validation split
# ============================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y.argmax(axis=1),
    random_state=SEED
)

# ============================================
# Model
# ============================================

input_dim = X.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),

    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(len(labels), activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

model.summary()

# ============================================
# Train
# ============================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# ============================================
# Save model + labels
# ============================================

os.makedirs("models", exist_ok=True)

model.save("models/landmark_model.h5")

with open("models/landmark_classes.txt", "w") as f:
    f.write("\n".join(labels))

print("Landmark model saved to models/landmark_model.h5")
print("Class labels saved to models/landmark_classes.txt")
