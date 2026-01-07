# -----------------------------
# Import required libraries
# -----------------------------
from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
import os

# -----------------------------
# Load trained model and class labels
# -----------------------------
model = load_model("models/model.h5")   # load the trained model
with open("models/classes.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]  # read class labels (A–Z, 0–9)

# -----------------------------
# Initialize MediaPipe Hands for hand detection
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,     # continuous video stream
    max_num_hands=2,             # detect up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Open webcam feed
# -----------------------------
cap = cv2.VideoCapture(0)

# Confidence threshold
CONF_THRESHOLD = 0.8
last_prediction = "..."

while True:
    ret, frame = cap.read()
    if not ret:
        break  # exit loop if frame not captured

    frame = cv2.flip(frame, 1)  # mirror effect (left-right flip)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR → RGB
    results = hands.process(frame_rgb)  # run MediaPipe hand detection

    if results.multi_hand_landmarks:
        # Initialize bounding box limits (normalized coords: 0–1)
        x_min, y_min = 1.0, 1.0
        x_max, y_max = 0.0, 0.0

        # Loop through detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            # Update bounding box for each landmark
            for lm in hand_landmarks.landmark:
                x_min = min(x_min, lm.x)
                y_min = min(y_min, lm.y)
                x_max = max(x_max, lm.x)
                y_max = max(y_max, lm.y)

            # Draw hand skeleton on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert normalized bounding box coords to pixel values, add padding
        pad = 40
        x1 = max(0, int(x_min * w) - pad)
        y1 = max(0, int(y_min * h) - pad)
        x2 = min(w, int(x_max * w) + pad)
        y2 = min(h, int(y_max * h) + pad)

        # Extract ROI (region of interest) if valid
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, (128, 128))     # resize to model input size
            roi_norm = roi_resized / 255.0                # normalize pixel values
            roi_input = np.expand_dims(roi_norm, axis=0)  # shape: (1,128,128,3)

            # Predict using model
            pred = model.predict(roi_input, verbose=0)[0]
            confidence = np.max(pred)
            class_idx = np.argmax(pred)
            class_label = class_labels[class_idx]

            # Apply confidence thresholding
            if confidence >= CONF_THRESHOLD:
                last_prediction = class_label

            # Display prediction on video frame (always show last stable)
            cv2.putText(frame, f"Prediction: {last_prediction} ({confidence:.2f})", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Draw ROI bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # Show extracted ROI in separate window
            cv2.imshow("ROI", roi_resized)

        else:
            # If invalid ROI, show blank image
            blank_roi = np.zeros((128,128,3), dtype=np.uint8)
            cv2.imshow("ROI", blank_roi)

    else:
        # If no hands detected, show blank ROI
        blank_roi = np.zeros((128,128,3), dtype=np.uint8)
        cv2.imshow("ROI", blank_roi)

    # Display webcam feed
    cv2.imshow("Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
hands.close()
