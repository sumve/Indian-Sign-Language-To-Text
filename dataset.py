import cv2
import mediapipe as mp
import numpy as np
import os
import time


# ==============================
# Configuration
# ==============================
DATASET_DIR = "DataSet"
IMG_SIZE = 224
SAVE_LANDMARKS = True

SAMPLES_PER_CLASS = 400
CHECKPOINT_INTERVAL = 100
SAVE_EVERY_N_FRAMES = 5
START_DELAY = 3

MIN_BOX_RATIO = 0.25                    # hand must fill at least 25% of output
MIN_BOX_SIZE = int(IMG_SIZE * MIN_BOX_RATIO)

print("RUNNING FILE:", __file__)
print("DATASET_DIR:", DATASET_DIR)


# ==============================
# MediaPipe setup
# ==============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ==============================
# Utility functions
# ==============================
def letterbox_resize(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_off = (size - nh) // 2
    x_off = (size - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas


def compute_union_bbox(all_landmarks, img_w, img_h):
    xs, ys = [], []
    for hand in all_landmarks:
        for lm in hand.landmark:
            xs.append(lm.x)
            ys.append(lm.y)

    cx = int(np.mean(xs) * img_w)
    cy = int(np.mean(ys) * img_h)

    palm_size = np.linalg.norm(
        np.array([
            all_landmarks[0].landmark[0].x - all_landmarks[0].landmark[9].x,
            all_landmarks[0].landmark[0].y - all_landmarks[0].landmark[9].y
        ])
    )

    box_size = int(palm_size * img_w * 3.2)

    x1 = max(0, cx - box_size // 2)
    y1 = max(0, cy - box_size // 2)
    x2 = min(img_w - 1, cx + box_size // 2)
    y2 = min(img_h - 1, cy + box_size // 2)

    return x1, y1, x2, y2


# ==============================
# Main loop
# ==============================
label = input("Enter class label: ").strip()

img_dir = os.path.join(DATASET_DIR, label)
lm_dir = os.path.join(DATASET_DIR, label + "_lm")

os.makedirs(img_dir, exist_ok=True)
if SAVE_LANDMARKS:
    os.makedirs(lm_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

count = len(os.listdir(img_dir))
frame_id = 0
capturing = False
paused = False

print(f"\nClass: {label}")
print("Press 's' to start | 'c' to continue after checkpoint | 'q' to quit")
time.sleep(START_DELAY)


while count < SAMPLES_PER_CLASS:
    ret, frame = cap.read()
    if not ret:
        break

    clean_frame = frame.copy()

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    roi = None
    too_small = False

    if res.multi_hand_landmarks:
        h, w, _ = frame.shape
        x1, y1, x2, y2 = compute_union_bbox(res.multi_hand_landmarks, w, h)

        box_w = x2 - x1
        box_h = y2 - y1
        too_small = box_w < MIN_BOX_SIZE or box_h < MIN_BOX_SIZE

        if not too_small:
            roi = clean_frame[y1:y2, x1:x2]

        # preview only
        for hand in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if too_small:
            cv2.putText(frame, "MOVE CLOSER",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and not capturing:
        print("Starting capture in 3 seconds...")
        time.sleep(3)
        capturing = True
        print("Capture started.")

    if capturing and not paused and roi is not None and frame_id % SAVE_EVERY_N_FRAMES == 0:
        roi_lb = letterbox_resize(roi, IMG_SIZE)
        cv2.imwrite(os.path.join(img_dir, f"{label}_{count:04d}.jpg"), roi_lb)

        if SAVE_LANDMARKS:
            lm = []
            for hand in res.multi_hand_landmarks:
                lm.append([[p.x, p.y, p.z] for p in hand.landmark])
            np.save(os.path.join(lm_dir, f"{label}_{count:04d}.npy"), np.array(lm))

        count += 1
        print(f"Saved {count}/{SAMPLES_PER_CLASS}")

        if count % CHECKPOINT_INTERVAL == 0:
            paused = True
            capturing = False
            print("\n--- CHECKPOINT ---")
            print("Change location / lighting. Press 'c' to continue.\n")

    if paused and key == ord('c'):
        paused = False
        capturing = False
        print("Ready. Press 's' to resume.")

    if key == ord('q'):
        break

    status = "WAITING" if not capturing else ("PAUSED" if paused else "CAPTURING")
    cv2.putText(frame,
                f"{label}: {count}/{SAMPLES_PER_CLASS} | {status}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if capturing else (0, 0, 255),
                2)

    cv2.imshow("Collecting", frame)
    frame_id += 1


cap.release()
cv2.destroyAllWindows()
hands.close()
print("Done.")
