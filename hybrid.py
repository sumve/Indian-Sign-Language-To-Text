import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model

# =========================
# Load models + labels
# =========================

img_model = load_model("models/model.h5")
lm_model  = load_model("models/landmark_model.h5")

with open("models/classes.txt") as f:
    class_labels = [l.strip() for l in f]

# =========================
# MediaPipe Hands
# =========================

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# Landmark utilities
# =========================

def normalize_hand(hand):
    wrist = hand[0]
    hand = hand - wrist
    palm_len = np.linalg.norm(hand[9]) + 1e-6
    return hand / palm_len

def landmarks_to_feature(multi_hand_landmarks):
    """
    Always returns (126,)
    """
    feats = []

    for hand_lms in multi_hand_landmarks[:2]:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        pts = normalize_hand(pts)
        feats.append(pts.reshape(-1))

    if len(feats) == 1:
        feats.append(np.zeros(63, dtype=np.float32))

    return np.concatenate(feats).astype(np.float32)

# =========================
# Webcam
# =========================

cap = cv2.VideoCapture(0)

CONF_LM_GATE = 0.80      # landmark confidence threshold
CONF_IMG_MIN = 0.60

last_pred = None
stable_count = 0
STABLE_FRAMES = 12
last_accept_time = 0
COOLDOWN = 1.5

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = None
    confidence = 0.0

    if result.multi_hand_landmarks:
        # ---------- LANDMARK MODEL ----------
        lm_feat = landmarks_to_feature(result.multi_hand_landmarks)
        lm_pred = lm_model.predict(lm_feat[np.newaxis, ...], verbose=0)[0]
        lm_idx  = np.argmax(lm_pred)
        lm_conf = float(np.max(lm_pred))

        # ---------- ROI for IMAGE MODEL ----------
        x_min, y_min = 1.0, 1.0
        x_max, y_max = 0.0, 0.0

        for hlm in result.multi_hand_landmarks:
            for lm in hlm.landmark:
                x_min = min(x_min, lm.x)
                y_min = min(y_min, lm.y)
                x_max = max(x_max, lm.x)
                y_max = max(y_max, lm.y)

            mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        pad = 40
        x1 = max(0, int(x_min * w) - pad)
        y1 = max(0, int(y_min * h) - pad)
        x2 = min(w, int(x_max * w) + pad)
        y2 = min(h, int(y_max * h) + pad)

        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (224, 224))
            roi = roi / 255.0
            roi = roi[np.newaxis, ...]

            img_pred = img_model.predict(roi, verbose=0)[0]
            img_idx  = np.argmax(img_pred)
            img_conf = float(np.max(img_pred))

            # ---------- HYBRID DECISION ----------
            if lm_conf >= CONF_LM_GATE:
                prediction = class_labels[lm_idx]
                confidence = lm_conf
            elif img_conf >= CONF_IMG_MIN:
                prediction = class_labels[img_idx]
                confidence = img_conf

    # ---------- STABILITY + COOLDOWN ----------
    if prediction == last_pred:
        stable_count += 1
    else:
        stable_count = 0

    last_pred = prediction

    if prediction and stable_count >= STABLE_FRAMES:
        if time.time() - last_accept_time > COOLDOWN:
            print("Accepted:", prediction, f"({confidence:.2f})")
            last_accept_time = time.time()
            stable_count = 0

    # ---------- UI ----------
    if prediction:
        cv2.putText(frame, f"{prediction} ({confidence:.2f})",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 2)

    cv2.imshow("Hybrid ISL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
