# Indian Sign Language (ISL) to Text Recognition

This project is a real-time **Sign Language to Text Conversion** system designed to improve accessibility for people who communicate using sign language. The system captures hand gestures through a webcam, recognizes the corresponding sign using deep learning & computer vision, and converts it into readable text. It also provides word suggestions and text-to-speech output to enhance usability.

The application is built as a desktop GUI and runs completely **offline**.The system is glove-free, runs on a standard webcam, and is optimized for **CPU-based real-time inference**.

---

## 🔍 Overview

This project detects one or two hands using MediaPipe, extracts a dynamic Region of Interest (ROI), and classifies the gesture using a **MobileNetV2-based CNN** trained on a **self-curated ISL dataset**.  
Stability and cooldown logic are applied to ensure accurate and readable text output.

---

## ✨ Features

- **47-class gesture recognition: Images & Landmarks(.npy)**
  - Alphabets: A–Z  
  - Digits: 0–9  
  - Control gestures: `SPACE`, `DELETE`, `CLEAR`
- Real-time webcam inference
- Multi-hand landmark-driven ROI extraction
- Gesture stabilization using consecutive-frame validation
- Cooldown policy to avoid repeated predictions
- Desktop GUI using Tkinter
- Offline **Text-to-Speech (TTS)** output using pTTS
- Complete pipeline: data collection → training → inference
- Hybrid prediction pipeline utilising both images & numpy arrays of hand landmarks for better accuracy.
- Stable prediction logic to avoid flickering outputs
- Sentence formation from continuous gestures
- Word suggestions to assist faster text completion
- Delete (backspace) and Clear controls for easy correction
- 300 images per class with two distinct people & varied lighting conditions.

---

## 🧠 Technology Stack

- **OpenCV** – Webcam capture and image processing  
- **MediaPipe Hands** – 21-point hand landmark detection (multi-hand support)  
- **TensorFlow / Keras** – Model training and inference  
- **MobileNetV2** – Transfer learning backbone  
- **AdamW + Cosine Decay** – Optimized training strategy  
- **scikit-learn** – Class balancing  
- **Tkinter** – Desktop GUI  
- **pTTS** – Text-to-speech synthesis  

---

## 📊 Dataset

- Self-collected static-frame ISL dataset
- ~300 images per class (≈ 14100 images total)
- 80/20 training–validation split
- Strong data augmentation:
  - Rotation, zoom, shear
  - Brightness variation
  - Horizontal flips
  - Normalization (1/255)

## Dataset Structure
```
dataSet/
├── A/
├── B/
├── C/
├── D/
├── E/
├── F/
├── G/
├── H/
├── I/
├── J/
├── K/
├── L/
├── M/
├── N/
├── O/
├── P/
├── Q/
├── R/
├── S/
├── T/
├── U/
├── V/
├── W/
├── X/
├── Y/
├── Z/
├── 0/
├── 1/
├── 2/
├── 3/
├── 4/
├── 5/
├── 6/
├── 7/
├── 8/
├── 9/
├── SPACE/
├── CLEAR/
├── DELETE/
├── Hello/
├── Thankyou/
├── Please/
├── Sorry/
├── Yes/
├── No/
├── Goodbye/
└── I Love You/
```

## Usage

### 1. Collect Gesture Data

Run the dataset collection script:

python new_dataset.py

- Automatically creates class folders inside `DataSet/`
- Captures padded hand ROIs using MediaPipe landmarks
- Saves resized gesture images & landmarks (class_lm) for each class

---

### 2. Train the Model

Train the classifier using transfer learning:

python train_new.py
python landmark_train.py

Training details:
- Transfer learning using MobileNetV2
- Strong data augmentation
- Class-weight balancing
- Progressive fine-tuning
- Early stopping and model checkpointing
- Best model saved as `models/model.h5`
- Best landmark model saved as `models/landmark_model.h5`
---

### 3. Run Real-Time Inference

Start real-time sign recognition:

python gui.py

- Webcam feed is mirrored
- Unified ROI across detected hands
- Prediction accepted only after stability and cooldown checks
- Output text displayed live in the GUI

---

### 4. Text-to-Speech Output

- Click the **Speak** button in the GUI to hear the detected text using pTTS

## Results

- Validation accuracy: ~98–99%
- Real-time inference speed: ~18–25 FPS on CPU
- Stable predictions for static ISL gestures
- Stability and cooldown logic significantly reduce false positives

## Project Structure

A breakdown of the scripts and directories included in this project:

```text
.
├── __pycache__/             # Python cache files
├── DataSet/                 # Photos & Landmarks Dataset
├── ISLData/                 # Indian Sign Language data (Kaggle)
├── models/                  # Active/latest trained models
├── models_old/              # Previous model iterations
├── new/                     # Current development directory
│   ├── check_lm.py          # Landmarks verification
│   ├── gui.py               # Graphical User Interface implementation
│   ├── hybrid_text.py       # Hybrid text processing logic
│   ├── hybrid.py            # Main hybrid model logic
│   ├── landmark_tra...      # Landmark training script
│   ├── new_dataset....      # Dataset preprocessing script
│   ├── train_new.py         # Updated training pipeline
├── old/                     # Legacy code/scripts
├── venv/                    # Primary virtual environment
├── venv2/                   # Alternative/testing virtual environment
├── .gitignore               # Files excluded from Git tracking
└── README.md                # Project documentation
```

## Future Improvements

- Expand the dataset to 10k+ images per class
- Add temporal modeling for dynamic and transition-based gestures
- Improve two-hand coordinated gesture recognition
- Optimize the model using quantization and pruning for mobile deployment
- Extend the system to full ISL sentence-level translation
- Integrate speech-to-sign and sign-to-speech bidirectional support

## License

This project is intended for academic, research, and portfolio use.

## Citation

If you use this project for research, coursework, or benchmarking, please reference this repository and include the gesture label mapping provided in `models/classes.txt` for reproducibility.
