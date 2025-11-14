import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import os

# Load history if saved

history_path = "models/history.pkl"
if os.path.exists(history_path):
    with open(history_path, "rb") as f:
        history = pickle.load(f)
else:
    print("History not saved during training. Cannot plot training stats.")
    history = None

if history:
    # Accuracy
    plt.figure(figsize=(10,5))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Loss
    plt.figure(figsize=(10,5))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
