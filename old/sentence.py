import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import tkinter as tk
from tkinter import messagebox
from gtts import gTTS
import os
from tensorflow.keras.models import load_model


# -----------------------------
# Load model and class labels
# -----------------------------

model = load_model("models/model.h5")
with open("models/classes.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# -----------------------------
# Initialize MediaPipe Hands
# -----------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,               # <--- allow 2 hands
    min_detection_confidence=0.5,  # same as your second script
    min_tracking_confidence=0.5
)

# -----------------------------
# Variables
# -----------------------------

display_text = ""
last_prediction = None
pad = 40
cooldown_time = 2.0
last_accept_time = 0
same_pred_count = 0
required_stability_frames = 15
current_word = ""
CONF_THRESHOLD = 0.0  # set >0 (e.g. 0.7) if you want confidence gating

# -----------------------------
# Function to play sound
# -----------------------------

def play_sound():
    global display_text
    if display_text.strip() == "":
        messagebox.showinfo("Info", "No text to speak!")
        return
    tts = gTTS(text=display_text, lang='en')
    audio_file = "output.mp3"
    tts.save(audio_file)
    try:
        if os.name == 'nt':
            os.startfile(audio_file)
        else:
            os.system(f"xdg-open {audio_file} &")
    except Exception:
        os.system(f"start {audio_file}")

# -----------------------------
# Tkinter UI
# -----------------------------

def start_tkinter_ui():
    root = tk.Tk()
    root.title("Sign to Text")

    lbl_text = tk.Label(root, text="Detected text will appear here", font=("Arial", 16))
    lbl_text.pack(pady=10)

    btn_play = tk.Button(root, text="Play Sound", font=("Arial", 14), command=play_sound)
    btn_play.pack(pady=10)

    def update_text():
        lbl_text.config(text=display_text)
        root.after(500, update_text)

    update_text()
    root.mainloop()

ui_thread = threading.Thread(target=start_tkinter_ui, daemon=True)
ui_thread.start()

# -----------------------------
# Webcam Loop
# -----------------------------

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    prediction = None

    if results.multi_hand_landmarks:
        # unified bounding box over ALL detected hands
        x_min, y_min = 1.0, 1.0
        x_max, y_max = 0.0, 0.0

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_min = min(x_min, lm.x)
                y_min = min(y_min, lm.y)
                x_max = max(x_max, lm.x)
                y_max = max(y_max, lm.y)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # convert to pixels and pad
        x1 = max(0, int(x_min * w) - pad)
        y1 = max(0, int(y_min * h) - pad)
        x2 = min(w, int(x_max * w) + pad)
        y2 = min(h, int(y_max * h) + pad)

        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, (128, 128))
            roi_norm = roi_resized / 255.0
            roi_input = np.expand_dims(roi_norm, axis=0)

            pred = model.predict(roi_input, verbose=0)[0]
            class_idx = np.argmax(pred)
            confidence = float(np.max(pred))
            prediction = class_labels[class_idx]

            # optional: ignore low-confidence predictions
            if confidence < CONF_THRESHOLD:
                prediction = None

            if prediction is not None:
                # stability logic
                if prediction == last_prediction:
                    same_pred_count += 1
                else:
                    same_pred_count = 0
                last_prediction = prediction

                # accept only after stable + cooldown
                if same_pred_count >= required_stability_frames and (time.time() - last_accept_time) > cooldown_time:
                    if prediction == "SPACE":
                        current_word += " "
                        print("Added SPACE")
                    elif prediction == "CLEAR":
                        current_word = ""
                        print("Cleared text")
                    elif prediction == "DELETE":
                        current_word = current_word[:-1]
                        print("Deleted last character")
                    else:
                        current_word += prediction
                        print(f"Accepted: {prediction} | Current word: {current_word}")

                    display_text = current_word
                    last_accept_time = time.time()
                    same_pred_count = 0  # reset after accepting

                    print(f"Accepted: {prediction} | Text: {display_text}")

            # draw ROI for debugging
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("ROI", roi_resized)

    else:
        same_pred_count = 0
        last_prediction = None
        prediction = None
        blank_roi = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.imshow("ROI", blank_roi)

    # -----------------------------
    # Overlay info on frame
    # -----------------------------
    cv2.putText(frame, f"Text: {display_text}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if prediction:
        cv2.putText(frame, f"Detected: {prediction}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign to Text (Camera)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Final text:", display_text)
