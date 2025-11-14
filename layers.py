from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("models/model.h5")

# Print summary of layers
model.summary()

# Optionally, list all layers with their types
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} - {layer.__class__.__name__}")
