from tkinter import Image
import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model

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
    hand = hand - hand[0]                     # wrist centered
    scale = np.linalg.norm(hand[9]) + 1e-6    # palm length
    return hand / scale

def preprocess_landmarks(all_hands):
    feats = []

    for hand_lms in all_hands[:2]:
        pts = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark],
                       dtype=np.float32)
        pts = normalize_hand(pts)
        feats.append(pts.reshape(-1))

    if len(feats) == 1:
        feats.append(np.zeros(63, dtype=np.float32))

    return np.concatenate(feats).reshape(1, 126)

# ===============================
# MAIN LOOP
# ===============================
cap = cv2.VideoCapture(0)
sentence = ""

last_pred = None
stable_count = 0
last_accept_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    final_pred = None
    image_conf = 0
    landmark_conf = 0

    if res.multi_hand_landmarks:
        h, w, _ = frame.shape

        # -------- UNION BBOX --------
        xs, ys = [], []
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
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---------- IMAGE MODEL ----------
        if roi.size > 0:
            img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            x_img = img.astype("float32") / 255.0
            x_img = np.expand_dims(x_img, axis=0)
            probs = image_model.predict(x_img, verbose=0)[0]
            idx = np.argmax(probs)
            image_conf = probs[idx]
            image_pred = CLASSES[idx]
        else:
            image_pred = None

        # ---------- LANDMARK MODEL ----------
        lm_input = preprocess_landmarks(res.multi_hand_landmarks)
        lm_probs = landmark_model.predict(lm_input, verbose=0)[0]
        lm_idx = np.argmax(lm_probs)
        landmark_conf = lm_probs[lm_idx]
        landmark_pred = CLASSES[lm_idx]

        # ---------- FUSION (image primary) ----------
        if image_conf >= IMAGE_CONF_THRESH:
            final_pred = image_pred
        if landmark_conf >= LANDMARK_CONF_THRESH:
            final_pred = landmark_pred

    # ---------- STABILITY ----------
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

    # ---------- UI ----------
    cv2.putText(frame, f"Sentence: {sentence}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if final_pred:
        cv2.putText(frame, f"Pred: {final_pred}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language To Text (Hybrid)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
