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
# User gesture directory
# -----------------------------

USER_GESTURE_DIR = "user_gestures"
os.makedirs(USER_GESTURE_DIR, exist_ok=True)

# -----------------------------
# Load model and labels
# -----------------------------

model = load_model("models/model.h5")
with open("models/classes.txt") as f:
    class_labels = [l.strip() for l in f]

# -----------------------------
# MediaPipe Hands
# -----------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Variables
# -----------------------------

sentence = ""
current_word = ""
display_text = ""

history = []
last_prediction = None
same_pred_count = 0
required_stability_frames = 15
cooldown_time = 2.0
last_accept_time = 0
pad = 40

USER_MODE = "BEGINNER"
CONF_THRESHOLD = 0.0

recording_gesture = False
recorded_landmarks = []
gesture_name = ""

fps = 0.0
frame_count = 0
start_time = time.time()
last_confidence = 0.0

# -----------------------------
# User mode
# -----------------------------

def apply_user_mode():
    global cooldown_time, required_stability_frames
    if USER_MODE == "BEGINNER":
        cooldown_time = 2.0
        required_stability_frames = 15
    else:
        cooldown_time = 0.8
        required_stability_frames = 7

apply_user_mode()

# -----------------------------
# Load user gestures
# -----------------------------

def load_user_gestures():
    gestures = {}
    for g in os.listdir(USER_GESTURE_DIR):
        lm_path = os.path.join(USER_GESTURE_DIR, g, "landmarks.npy")
        meta_path = os.path.join(USER_GESTURE_DIR, g, "meta.txt")
        if os.path.exists(lm_path) and os.path.exists(meta_path):
            ref = np.mean(np.load(lm_path), axis=0)
            with open(meta_path) as f:
                mapped = f.read().split("=")[1]
            gestures[g] = (ref, mapped)
    return gestures

user_gestures = load_user_gestures()

# -----------------------------
# Audio
# -----------------------------

def play_sound():
    if display_text.strip():
        gTTS(display_text).save("output.mp3")
        os.startfile("output.mp3")

# -----------------------------
# Export
# -----------------------------

def export_session():
    text = (sentence + current_word).strip()
    if not text:
        return
    os.makedirs("exports", exist_ok=True)
    path = time.strftime("exports/session_%Y%m%d_%H%M%S.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Exported:", path)

# -----------------------------
# Tkinter UI
# -----------------------------

def start_ui():
    root = tk.Tk()
    root.title("Sign to Text")
    lbl = tk.Label(root, font=("Arial", 16))
    lbl.pack(pady=10)
    tk.Button(root, text="Play Sound", command=play_sound).pack()
    def update():
        lbl.config(text=display_text)
        root.after(500, update)
    update()
    root.mainloop()

threading.Thread(target=start_ui, daemon=True).start()

# -----------------------------
# Webcam loop
# -----------------------------

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    prediction = None

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # Landmark vector
        lm_vec = []
        for lm in hand.landmark:
            lm_vec.extend([lm.x, lm.y, lm.z])
        lm_vec = np.array(lm_vec)

        # Recording mode
        if recording_gesture:
            recorded_landmarks.append(lm_vec)
            if len(recorded_landmarks) >= 25:
                path = os.path.join(USER_GESTURE_DIR, gesture_name)
                os.makedirs(path, exist_ok=True)
                np.save(os.path.join(path, "landmarks.npy"), np.array(recorded_landmarks))
                with open(os.path.join(path, "meta.txt"), "w") as f:
                    f.write(f"mapped_text={gesture_name}")
                recording_gesture = False
                user_gestures = load_user_gestures()
                print("Gesture recorded:", gesture_name)

        # Custom gesture match
        for _, (ref, mapped) in user_gestures.items():
            if np.linalg.norm(lm_vec - ref) < 0.15:
                prediction = mapped
                last_confidence = 1.0
                break

        # CNN fallback
        if prediction is None:
            x = [lm.x for lm in hand.landmark]
            x1, x2 = int(min(x) * w) - pad, int(max(x) * w) + pad
            y = [lm.y for lm in hand.landmark]
            y1, y2 = int(min(y) * h) - pad, int(max(y) * h) + pad
            # Ensure valid ROI before resizing
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]

                if roi.size != 0:
                    roi = cv2.resize(roi, (128, 128))
                    roi_norm = roi / 255.0
                    roi_input = np.expand_dims(roi_norm, axis=0)

                    pred = model.predict(roi_input, verbose=0)[0]
                    class_idx = np.argmax(pred)
                    prediction = class_labels[class_idx]
                    last_confidence = float(np.max(pred))
                else:
                    prediction = None
            else:
                prediction = None

            pred = model.predict(np.expand_dims(roi / 255.0, 0), verbose=0)[0]
            prediction = class_labels[np.argmax(pred)]
            last_confidence = float(np.max(pred))

        # Stability + commit
        if prediction == last_prediction:
            same_pred_count += 1
        else:
            same_pred_count = 0
        last_prediction = prediction

        if same_pred_count >= required_stability_frames and time.time() - last_accept_time > cooldown_time:
            history.append((sentence, current_word))
            if prediction == "SPACE":
                sentence += current_word + " "
                current_word = ""
            elif prediction == "DELETE":
                current_word = current_word[:-1]
            elif prediction == "CLEAR":
                sentence, current_word = "", ""
            else:
                current_word += prediction
            display_text = sentence + current_word
            last_accept_time = time.time()
            same_pred_count = 0

        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # FPS
    frame_count += 1
    if time.time() - start_time >= 1:
        fps = frame_count
        frame_count = 0
        start_time = time.time()

    # Overlays
    cv2.putText(frame, f"Sentence: {sentence}", (10, 100), 0, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Current: {current_word}", (10, 140), 0, 0.8, (200,200,200), 2)
    cv2.putText(frame, f"FPS: {fps}", (10, 180), 0, 0.7, (0,255,255), 2)

    if recording_gesture:
        cv2.putText(frame, "Recording gesture...", (10, 220), 0, 0.7, (0,0,255), 2)

    cv2.imshow("Sign to Text", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('m'):
        USER_MODE = "EXPERT" if USER_MODE == "BEGINNER" else "BEGINNER"
        apply_user_mode()
    if key == ord('e'):
        export_session()
    if key == ord('r') and not recording_gesture:
        gesture_name = input("New gesture name: ").strip()
        recorded_landmarks = []
        recording_gesture = True

cap.release()
cv2.destroyAllWindows()
hands.close()
