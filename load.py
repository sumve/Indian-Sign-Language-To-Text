from tensorflow.keras.models import load_model
model = load_model("models/landmark_model.h5")
model.summary()
