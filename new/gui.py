import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk
from PIL import Image, ImageTk
import threading
import pyttsx3
from tensorflow.keras.models import load_model
from wordfreq import top_n_list

# ===============================
# CONFIG
# ===============================
IMAGE_MODEL_PATH = "models/model.h5"
LANDMARK_MODEL_PATH = "models/landmark_model.h5"
CLASSES_PATH = "models/classes.txt"

IMG_SIZE = 224
IMAGE_CONF_THRESH = 0.75
LANDMARK_CONF_THRESH = 0.85

STABLE_FRAMES = 12
COOLDOWN_TIME = 0.8

SPACE_TOKEN = "Space"
CLEAR_TOKEN = "Clear"

# ===============================
# LOAD MODELS
# ===============================
image_model = load_model(IMAGE_MODEL_PATH)
landmark_model = load_model(LANDMARK_MODEL_PATH)

with open(CLASSES_PATH) as f:
    CLASSES = [c.strip() for c in f.readlines()]

# ===============================
# TEXT TO SPEECH
# ===============================
tts_engine = pyttsx3.init("sapi5")
tts_engine.setProperty("rate", 160)
tts_engine.setProperty("volume", 1.0)
voices = tts_engine.getProperty("voices")
tts_engine.setProperty("voice", voices[1].id)

# ===============================
# MEDIAPIPE
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ===============================
# LANDMARK UTILS
# ===============================
def normalize_hand(hand):
    hand = hand - hand[0]
    scale = np.linalg.norm(hand[9]) + 1e-6
    return hand / scale

def preprocess_landmarks(all_hands):
    feats = []
    for hand_lms in all_hands[:2]:
        pts = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark], dtype=np.float32)
        pts = normalize_hand(pts)
        feats.append(pts.reshape(-1))
    if len(feats) == 1:
        feats.append(np.zeros(63, dtype=np.float32))
    return np.concatenate(feats).reshape(1, 126)

# ===============================
# SUGGESTIONS (NO DICTIONARY)
# ===============================
COMMON_WORDS = top_n_list("en", 50000)

def get_current_word(text):
    return text.split(" ")[-1] if text.strip() else ""

def get_suggestions(prefix, limit=4):
    if not prefix:
        return []
    prefix = prefix.lower()
    out = []
    for w in COMMON_WORDS:
        if w.startswith(prefix):
            out.append(w.upper())
            if len(out) == limit:
                break
    return out

# ===============================
# GUI SETUP
# ===============================
root = tk.Tk()
root.title("Sign Language To Text & Speech Conversion")
root.geometry("1000x700")

tk.Label(root, text="Sign Language To Text & Speech Conversion",
         font=("Courier", 22, "bold")).pack(pady=10)

cam_label = tk.Label(root)
cam_label.pack()
info_frame = tk.Frame(root)
info_frame.pack(pady=20)
char_var = tk.StringVar(value="Character : ")
sent_var = tk.StringVar(value="Sentence : ")

tk.Label(
    info_frame,
    textvariable=char_var,
    font=("Courier", 16)
).pack(pady=5)

tk.Label(
    info_frame,
    textvariable=sent_var,
    font=("Courier", 16)
).pack(pady=5)


# ---------- Suggestions UI ----------
# ---------- Word suggestions row ----------
suggestion_row = tk.Frame(info_frame)
suggestion_row.pack(pady=(15, 5))

tk.Label(
    suggestion_row,
    text="Word Suggestions:",
    font=("Courier", 14, "bold")
).pack(side="left", padx=(0, 10))

suggestion_frame = tk.Frame(suggestion_row)
suggestion_frame.pack(side="left")
suggestion_buttons = []

sentence = ""

def clear_sentence():
    global sentence
    sentence = ""
    sent_var.set("Sentence : ")
    update_suggestions()

def delete_last_char():
    global sentence
    if sentence:
        sentence = sentence[:-1]
        sent_var.set(f"Sentence : {sentence}")
        update_suggestions()

def speak_sentence():
    if not sentence.strip():
        return
    threading.Thread(
        target=lambda: (tts_engine.say(sentence), tts_engine.runAndWait()),
        daemon=True
    ).start()

def apply_suggestion(word):
    global sentence
    parts = sentence.rstrip().split(" ")
    parts[-1] = word
    sentence = " ".join(parts) + " "
    sent_var.set(f"Sentence : {sentence}")
    update_suggestions()

def update_suggestions():
    for b in suggestion_buttons:
        b.destroy()
    suggestion_buttons.clear()

    current = get_current_word(sentence)
    for w in get_suggestions(current):
        btn = tk.Button(
            suggestion_frame,
            text=w,
            width=10,
            command=lambda x=w: apply_suggestion(x)
        )
        btn.grid(row=0, column=len(suggestion_buttons), padx=5)
        suggestion_buttons.append(btn)

btn_frame = tk.Frame(info_frame)
btn_frame.pack(pady=20)

tk.Button(
    btn_frame,
    text="Delete",
    width=10,
    command=delete_last_char
).grid(row=0, column=2, padx=10)

tk.Button(btn_frame, text="Clear", width=10, command=clear_sentence)\
    .grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Speak", width=10, command=speak_sentence)\
    .grid(row=0, column=1, padx=10)

# ===============================
# VIDEO STATE
# ===============================
cap = cv2.VideoCapture(0)
last_pred = None
stable_count = 0
last_accept_time = 0

# ===============================
# MAIN LOOP
# ===============================
def update():
    global last_pred, stable_count, last_accept_time, sentence

    ret, frame = cap.read()
    if not ret:
        root.after(10, update)
        return

    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    final_pred = None
    confidence = 0.0

    if res.multi_hand_landmarks:
        xs, ys = [], []
        h, w, _ = frame.shape

        for hand in res.multi_hand_landmarks:
            for lm in hand.landmark:
                xs.append(lm.x)
                ys.append(lm.y)

        x1 = max(0, int(min(xs) * w) - 40)
        y1 = max(0, int(min(ys) * h) - 40)
        x2 = min(w, int(max(xs) * w) + 40)
        y2 = min(h, int(max(ys) * h) + 40)

        roi = frame[y1:y2, x1:x2]

        for hand in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255,255,255), thickness=2)
            )

        img_conf = 0
        if roi.size > 0:
            img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            probs = image_model.predict(
                np.expand_dims(img / 255.0, axis=0), verbose=0
            )[0]
            img_conf = probs.max()
            img_pred = CLASSES[np.argmax(probs)]

        lm_probs = landmark_model.predict(
            preprocess_landmarks(res.multi_hand_landmarks), verbose=0
        )[0]
        lm_conf = lm_probs.max()
        lm_pred = CLASSES[np.argmax(lm_probs)]

        if lm_conf >= LANDMARK_CONF_THRESH:
            final_pred, confidence = lm_pred, lm_conf
        elif img_conf >= IMAGE_CONF_THRESH:
            final_pred, confidence = img_pred, img_conf

        if final_pred:
            cv2.putText(
                frame,
                f"Gesture: {final_pred}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),   # BLACK
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence*100:.1f}%",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),   # BLACK
                2
            )


    if final_pred == last_pred:
        stable_count += 1
    else:
        stable_count = 0
    last_pred = final_pred

    if final_pred and stable_count >= STABLE_FRAMES:
        if time.time() - last_accept_time > COOLDOWN_TIME:
            last_accept_time = time.time()
            stable_count = 0

            if final_pred == SPACE_TOKEN:
                if sentence and not sentence.endswith(" "):
                    sentence += " "
            elif final_pred == CLEAR_TOKEN:
                sentence = ""
            else:
                sentence += final_pred

            char_var.set(f"Detected Gesture : {final_pred}")
            sent_var.set(f"Sentence : {sentence}")
            update_suggestions()

    img = ImageTk.PhotoImage(Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ))
    cam_label.imgtk = img
    cam_label.configure(image=img)
    info_frame = tk.Frame(root)
    info_frame.pack(pady=20)


    root.after(10, update)

# ===============================
# START
# ===============================
update()
root.mainloop()
cap.release()
hands.close()
