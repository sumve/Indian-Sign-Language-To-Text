import cv2
import os
import time
import mediapipe as mp

def collect_data(label, save_dir, num_samples=100, save_every=1):
    """
    Args:
        label (str): The gesture label (e.g., "A", "thankyou", "please").
        save_dir (str): Path where images will be stored (e.g., "dataSet/thankyou").
        num_samples (int): Number of images to capture.
        save_every (int): Save every nth frame to reduce duplicates.
    """

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    os.makedirs(save_dir, exist_ok=True)
    count = len(os.listdir(save_dir))

    print(f"Ready to collect images for '{label}' into {save_dir}.")
    print("Press 's' to start, 'q' to quit.")

    capturing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror preview
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        roi = None

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = 1.0, 1.0, 0.0, 0.0

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_min, y_min = min(x_min, lm.x), min(y_min, lm.y)
                    x_max, y_max = max(x_max, lm.x), max(y_max, lm.y)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x1, y1 = int(x_min * w), int(y_min * h)
            x2, y2 = int(x_max * w), int(y_max * h)

            pad = 40
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    roi = cv2.resize(roi, (128, 128))
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Frame", frame)
        if roi is not None:
            cv2.imshow("ROI", roi)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and not capturing:
            print("Starting capture in 3 seconds...")
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            print("Go!")
            capturing = True

        if capturing and roi is not None:
            if count % save_every == 0:
                img_path = os.path.join(save_dir, f"{label}_{count}.jpg")
                cv2.imwrite(img_path, roi)
                print(f"Saved: {img_path}")
            count += 1

        if key == ord('q') or (capturing and count >= num_samples):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"Finished collecting {count} images for '{label}'.")


if __name__ == "__main__":
    label = input("Enter the gesture name (A–Z, 0–9, or custom like 'thankyou'): ").strip()
    save_dir = f"dataSet/{label}"
    collect_data(label, save_dir, num_samples=100, save_every=1)
