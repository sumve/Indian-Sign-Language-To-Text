# -----------------------------
# Import required libraries
# -----------------------------
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from sklearn.utils import class_weight
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers import AdamW
import numpy as np
import pickle

# -----------------------------
# 1. Paths and constants
# -----------------------------
DATASET_DIR = "C:/SumV/SignLanguageToText/dataSet"  # directory containing class folders
IMG_SIZE = (128, 128)  # input size for model
BATCH_SIZE = 64        # number of images per training batch

# -----------------------------
# 2. Data Generators (with augmentation)
# -----------------------------
# Create a generator that applies strong augmentation to training images
train_datagen = ImageDataGenerator(
    rescale=1./255,             # normalize pixel values
    rotation_range=10,          # random rotations
    zoom_range=0.15,             # random zoom
    width_shift_range=0.1,     # random horizontal shift
    height_shift_range=0.1,    # random vertical shift
    shear_range=0.1,           # random shear
    brightness_range=[0.6, 1.4],# vary brightness
    horizontal_flip=True,       # allow horizontal flips
    validation_split=0.2        # split dataset into training and validation
)

# Load training subset
train_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical",
    color_mode="rgb"
)

# Load validation subset
val_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical",
    color_mode="rgb"
)

# Get class labels from dataset
labels = list(train_data.class_indices.keys())
print("Classes detected:", labels)
print("Number of classes detected:", len(labels))

# Ensure dataset contains all expected classes (26 letters + 10 digits = 36)
expected_classes = 26 + 10
if len(labels) != expected_classes:
    print(f"WARNING: Expected {expected_classes} classes, but found {len(labels)}.")
    print("Check your dataset: at least one folder may be empty or misnamed.")

# -----------------------------
# 3. Compute Class Weights
# -----------------------------
# Balance dataset so minority classes are not ignored
classes = train_data.classes
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(classes),
    y=classes
)
class_weights = dict(enumerate(class_weights))

# -----------------------------
# 4. Transfer Learning with MobileNetV2
# -----------------------------
# Load pre-trained MobileNetV2 (without top classifier layer)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze base model initially

# Build classification head on top of MobileNetV2
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # reduce spatial dimensions to vector

    # Fully connected layers with regularization + dropout
    layers.Dense(512, activation="relu",
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(256, activation="relu",
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Output layer (softmax for multi-class classification)
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.summary()

# -----------------------------
# 5. Callbacks (training helpers)
# -----------------------------
# Save best model based on validation accuracy
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# Stop training if validation loss stops improving
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when validation accuracy plateaus
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# -----------------------------
# 6. Train (Stage 1: Frozen base)
# -----------------------------
initial_lr = 1e-3
lr_schedule = CosineDecay(initial_learning_rate=initial_lr, decay_steps=1000)
optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

EPOCHS = 10
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# -----------------------------
# 7. Progressive Fine-tuning
# -----------------------------
# Function to unfreeze deeper layers of base model in stages
def fine_tune_model(model, base_model, unfreeze_from, epochs=3):
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False  # keep earlier layers frozen

    # Lower learning rate for fine-tuning
    fine_tune_lr = 1e-5
    lr_schedule_fine = CosineDecay(initial_learning_rate=fine_tune_lr, decay_steps=500)
    optimizer_fine = AdamW(learning_rate=lr_schedule_fine, weight_decay=1e-5)

    model.compile(optimizer=optimizer_fine,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history_fine = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    return history_fine

# Fine-tuning in multiple stages:
# Stage 1: Train last 20 layers
history_fine1 = fine_tune_model(model, base_model,
                                unfreeze_from=len(base_model.layers)-20,
                                epochs=4)

# Stage 2: Train last 50 layers
history_fine2 = fine_tune_model(model, base_model,
                                unfreeze_from=len(base_model.layers)-50,
                                epochs=4)

# Stage 3: Train entire base model
history_fine3 = fine_tune_model(model, base_model,
                                unfreeze_from=0,
                                epochs=4)


# Save training history
with open("models/history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# -----------------------------
# 8. Save class labels for later use
# -----------------------------
os.makedirs("models", exist_ok=True)
labels = list(train_data.class_indices.keys())
with open("models/classes.txt", "w") as f:
    f.write("\n".join(labels))

print("Training complete. Best model saved as model.h5")
print("Classes:", labels)
