# -----------------------------
# Import required libraries
# -----------------------------
import os
import tensorflow as tf
import numpy as np
import pickle
from collections import Counter

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers import AdamW
from sklearn.utils import class_weight

# -----------------------------
# 1. Paths and constants
# -----------------------------
DATASET_DIR = "C:/SumV/SignLanguageToText/DataSet"  # FIXED casing
IMG_SIZE = (224, 224)                              # match collection
BATCH_SIZE = 64
EPOCHS = 10

print("Training from:", DATASET_DIR)

# -----------------------------
# 2. Data Generators
# -----------------------------
# Training generator (augmentation ON)
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.7, 1.3],
    validation_split=0.2
)

# Validation generator (NO augmentation)
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical",
    color_mode="rgb"
)

val_data = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical",
    color_mode="rgb"
)

labels = list(train_data.class_indices.keys())
num_classes = train_data.num_classes

print("Classes detected:", labels)
print("Number of classes:", num_classes)

# -----------------------------
# 3. Class distribution + weights
# -----------------------------
class_counts = Counter(train_data.classes)
print("Class distribution:", class_counts)

class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# -----------------------------
# 4. Model: MobileNetV2 backbone
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),

    layers.Dense(512, activation="relu",
                 kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(256, activation="relu",
                 kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    layers.Dense(num_classes, activation="softmax")
])

model.summary()

# -----------------------------
# 5. Callbacks
# -----------------------------
os.makedirs("models", exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# -----------------------------
# 6. Stage 1: Train classifier head
# -----------------------------
steps_per_epoch = train_data.samples // BATCH_SIZE
decay_steps = steps_per_epoch * EPOCHS

lr_schedule = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=decay_steps
)

optimizer = AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-4
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

history_initial = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# -----------------------------
# 7. Progressive Fine-tuning
# -----------------------------
def fine_tune_model(unfreeze_from, epochs):
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    steps = train_data.samples // BATCH_SIZE
    lr_schedule_fine = CosineDecay(
        initial_learning_rate=1e-5,
        decay_steps=steps * epochs
    )

    optimizer_fine = AdamW(
        learning_rate=lr_schedule_fine,
        weight_decay=1e-5
    )

    model.compile(
        optimizer=optimizer_fine,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )

    return model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

history_fine1 = fine_tune_model(len(base_model.layers) - 20, epochs=4)
history_fine2 = fine_tune_model(len(base_model.layers) - 50, epochs=4)
history_fine3 = fine_tune_model(0, epochs=4)

# -----------------------------
# 8. Save training history
# -----------------------------
full_history = {
    "initial": history_initial.history,
    "fine_20": history_fine1.history,
    "fine_50": history_fine2.history,
    "fine_all": history_fine3.history
}

with open("models/history.pkl", "wb") as f:
    pickle.dump(full_history, f)

# -----------------------------
# 9. Save class labels
# -----------------------------
with open("models/classes.txt", "w") as f:
    f.write("\n".join(labels))

print("Training complete.")
print("Best model saved as models/model.h5")
print("Classes:", labels)
