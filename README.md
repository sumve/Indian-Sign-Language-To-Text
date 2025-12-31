# Indian Sign Language (ISL) to Text Recognition

A real-time **Indian Sign Language (ISL) to Text** recognition system that translates hand gestures into readable text using computer vision and deep learning.  
The system is **glove-free**, runs on a standard webcam, and is optimized for **CPU-based real-time inference**.

---

## 🔍 Overview

This project detects one or two hands using MediaPipe, extracts a dynamic Region of Interest (ROI), and classifies the gesture using a **MobileNetV2-based CNN** trained on a **self-curated ISL dataset**.  
Stability and cooldown logic are applied to ensure accurate and readable text output.

---

## ✨ Features

- **49-class gesture recognition**
  - Alphabets: A–Z  
  - Digits: 0–9  
  - Control gestures: `SPACE`, `DELETE`, `CLEAR`, `DONE`, `NEXT`
- Real-time webcam inference
- Multi-hand landmark-driven ROI extraction
- Gesture stabilization using consecutive-frame validation
- Cooldown policy to avoid repeated predictions
- Desktop GUI using Tkinter
- Optional **Text-to-Speech (TTS)** output using gTTS
- Complete pipeline: data collection → training → inference

---

## 🧠 Technology Stack

- **OpenCV** – Webcam capture and image processing  
- **MediaPipe Hands** – 21-point hand landmark detection (multi-hand support)  
- **TensorFlow / Keras** – Model training and inference  
- **MobileNetV2** – Transfer learning backbone  
- **AdamW + Cosine Decay** – Optimized training strategy  
- **scikit-learn** – Class balancing  
- **Tkinter** – Desktop GUI  
- **gTTS** – Text-to-speech synthesis  

---

## 📊 Dataset

- Self-collected static-frame ISL dataset
- ~100 images per class (≈ 4900 images total)
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
├── DONE/
├── NEXT/
├── Hello/
├── Thankyou/
├── Please/
├── Sorry/
├── Yes/
├── No/
└── ILY/
```

## Usage

### 1. Collect Gesture Data

Run the dataset collection script:

python collect_data.py

- Automatically creates class folders inside `dataSet/`
- Captures padded hand ROIs using MediaPipe landmarks
- Saves resized gesture images for each class

---

### 2. Train the Model

Train the classifier using transfer learning:

python train.py

Training details:
- Transfer learning using MobileNetV2
- Strong data augmentation
- Class-weight balancing
- Progressive fine-tuning
- Early stopping and model checkpointing
- Best model saved as `models/model.h5`

---

### 3. Run Real-Time Inference

Start real-time sign recognition:

python webcam.py

- Webcam feed is mirrored
- Unified ROI across detected hands
- Prediction accepted only after stability and cooldown checks
- Output text displayed live in the GUI

---

### 4. Text-to-Speech Output

- Click the **Play Sound** button in the GUI to hear the detected text using TTS

## Results

- Validation accuracy: ~98–99%
- Real-time inference speed: ~18–25 FPS on CPU
- Stable predictions for static ISL gestures
- Majority of errors occur between visually similar hand shapes
- Stability and cooldown logic significantly reduce false positives

## Project Structure

A breakdown of the scripts and directories included in this project:

```text
.
├── collect_data.py       # Dataset collection script
├── train.py              # Model training script
├── webcam.py             # Real-time inference + GUI
├── plot.py               # Training curves visualization
├── layers.py             # Model architecture inspection
├── count.py              # Dataset class distribution check
├── folders.py            # Dataset folder initialization
├── models/               # Directory for saved models
│   ├── model.h5          # Trained model weights
│   └── classes.txt       # Class label mapping
├── dataSet/              # Gesture dataset (A–Z, 0–9, control gestures)
└── README.md             # Project documentation
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
