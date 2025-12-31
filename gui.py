import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk
from PIL import Image, ImageTk
from gtts import gTTS
import os
from tensorflow.keras.models import load_model

# =============================
# Load model and labels
# =============================

model = load_model("models/model.h5")
with open("models/classes.txt") as f:
    class_labels = [l.strip() for l in f]

# =============================
# MediaPipe Hands
# =============================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =============================
# Global State
# =============================

sentence = ""
current_word = ""
display_text = ""
last_prediction = None
same_pred_count = 0
required_stability_frames = 15
cooldown_time = 2.0
last_accept_time = 0
pad = 40

fps = 0
frame_count = 0
start_time = time.time()

# =============================
# Audio
# =============================

def play_sound():
    if display_text.strip():
        gTTS(display_text).save("output.mp3")
        os.startfile("output.mp3")

def clear_text():
    global sentence, current_word, display_text
    sentence = ""
    current_word = ""
    display_text = ""

# =============================
# Tkinter GUI
# =============================

root = tk.Tk()
root.title("Sign Language To Text Conversion")
root.geometry("1200x800")

title = tk.Label(
    root,
    text="Sign Language To Text Conversion",
    font=("Arial", 22, "bold")
)
title.pack(pady=10)

main_frame = tk.Frame(root)
main_frame.pack()

cam_label = tk.Label(main_frame)
cam_label.grid(row=0, column=0, padx=10)

skeleton_label = tk.Label(main_frame)
skeleton_label.grid(row=0, column=1, padx=10)

output_frame = tk.Frame(root)
output_frame.pack(pady=10)

char_var = tk.StringVar(value="Character:")
sent_var = tk.StringVar(value="Sentence:")

tk.Label(output_frame, textvariable=char_var, font=("Arial", 14)).pack(anchor="w")
tk.Label(output_frame, textvariable=sent_var, font=("Arial", 14)).pack(anchor="w")

btn_frame = tk.Frame(output_frame)
btn_frame.pack(pady=5)

tk.Button(btn_frame, text="Clear", width=10, command=clear_text).pack(side="left", padx=5)
tk.Button(btn_frame, text="Speak", width=10, command=play_sound).pack(side="left", padx=5)

# =============================
# Skeleton Canvas
# =============================

def get_skeleton_frame(results, w=480, h=480):
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                canvas,
                hand,
                mp_hands.HAND_CONNECTIONS
            )
    return canvas

# =============================
# Webcam + ML Loop (GUI-safe)
# =============================

cap = cv2.VideoCapture(0)

def update_gui():
    global last_prediction, same_pred_count, display_text
    global frame_count, fps, start_time, last_accept_time
    global sentence, current_word

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_gui)
        return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    prediction = None

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]

        # center of hand
        cx = int(np.mean(xs) * w)
        cy = int(np.mean(ys) * h)

        # square size based on hand span
        size = int(
            max(
                (max(xs) - min(xs)) * w,
                (max(ys) - min(ys)) * h
            )
        ) + 2 * pad

        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w, cx + size // 2)
        y2 = min(h, cy + size // 2)


        if prediction == last_prediction:
            same_pred_count += 1
        else:
            same_pred_count = 0
        last_prediction = prediction

        if (
            same_pred_count >= required_stability_frames and
            time.time() - last_accept_time > cooldown_time and
            prediction is not None
        ):
            if prediction == "SPACE":
                sentence += current_word + " "
                current_word = ""
            elif prediction == "DELETE":
                current_word = current_word[:-1]
            elif prediction == "CLEAR":
                sentence = ""
                current_word = ""
            else:
                current_word += prediction

            display_text = sentence + current_word
            last_accept_time = time.time()
            same_pred_count = 0

    # ================= FPS =================
    frame_count += 1
    if time.time() - start_time >= 1:
        fps = frame_count
        frame_count = 0
        start_time = time.time()

    # ================= GUI Updates =================

    cam_img = ImageTk.PhotoImage(Image.fromarray(rgb))
    cam_label.imgtk = cam_img
    cam_label.configure(image=cam_img)

    skeleton = get_skeleton_frame(results)
    skel_img = ImageTk.PhotoImage(Image.fromarray(skeleton))
    skeleton_label.imgtk = skel_img
    skeleton_label.configure(image=skel_img)

    char_var.set(f"Character: {last_prediction or ''}")
    sent_var.set(f"Sentence: {display_text}")

    root.after(10, update_gui)

# =============================
# Start
# =============================

update_gui()
root.mainloop()

cap.release()
hands.close()
