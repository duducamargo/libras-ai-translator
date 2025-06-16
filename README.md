
# 🤖 Libras Alphabet Recognition with AI

## 📚 Overview

This project is a complete pipeline for recognizing hand gestures representing letters of the **Brazilian Sign Language (Libras)** using real-time video input, computer vision, and deep learning.

It includes a fully custom dataset, model training using CNNs, and real-time prediction using webcam input. The goal is to offer a reproducible and scalable foundation for gesture recognition in educational or assistive contexts.

---

## 🎯 Objectives

- Capture hand gestures for selected Libras alphabet letters (currently A, B, C, L, M, N, O, Q, S, U, V, W)
- Build and train a convolutional neural network (CNN) using custom images (32x32 RGB)
- Predict letters in real time using the webcam, with hand segmentation and confidence scoring
- Maintain a modular, clean and extensible project architecture

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **OpenCV** - camera access, ROI extraction, frame processing
- **MediaPipe** - hand detection and background segmentation
- **TensorFlow / Keras** - CNN model training and live prediction
- **Jupyter Notebook** - model training & validation workflow
- **NumPy** - image normalization and matrix operations

---

## 🧱 Project Structure

```
libras-recognition/
├── dataset/                # Images organized by letter (excluded from git)
├── models/                 # Trained model + class mapping
│   ├── asl_gesture_model2.h5
│   └── class_indices.json
├── src/
│   ├── camera.py           # Real-time prediction
│   ├── coletar_dataset.py  # Data capture via webcam
│   └── __init__.py
├── notebooks/
│   ├── training.ipynb      # CNN training
│   └── validate.ipynb      # Accuracy and predictions check
├── requirements.txt
└── README.md
```

---

## 📸 Dataset Collection

The script `coletar_dataset.py` enables the user to collect custom hand gesture images via webcam. It uses MediaPipe for hand detection and segmentation (black background).

- Press any key from `a` to `z` to start capturing that letter.
- Images are saved in `dataset/<letter>/`.
- Resolution: 32x32 RGB.
- Format: `.jpg`

> Example folder: `dataset/a/` will contain `a_0000.jpg`, `a_0001.jpg`, ...

---

## 🧠 Model Training

- Images are loaded and processed via `ImageDataGenerator` in `training.ipynb`.
- Input shape: (32, 32, 3)
- Output: softmax classification across all gesture classes.
- Model saved to `models/asl_gesture_model2.h5`
- Class mapping saved to `models/class_indices.json` like:  
  `{"a": 0, "b": 1, "c": 2, ...}`

> The CNN is trained using a simple architecture with two Conv2D layers and Dropout.

---

## 🎥 Real-Time Prediction

The script `camera.py` loads the trained model and class mapping to predict the user's hand gesture live.

**Key Features:**
- Segments the background using MediaPipe.
- Detects a single hand and crops the region of interest (ROI).
- Normalizes the image and feeds it to the model.
- Displays prediction and confidence score live on screen.
- Green text for high confidence (≥ 0.75), red otherwise.

---

## ✅ Completed Milestones

- [x] Created custom dataset from scratch using webcam
- [x] Designed and trained CNN model using TensorFlow/Keras
- [x] Implemented real-time prediction script
- [x] Removed dataset from Git tracking and added `.gitignore`
- [x] Restructured the project into modular folders

---

## 🚀 Next Steps

- Expand dataset to full alphabet (A–Z)
- Add more training samples per class for robustness
- Improve model architecture (e.g., use more advanced CNNs)
- Deploy with GUI, Streamlit, or Flask web interface
- Export to mobile-friendly format (e.g., TensorFlow Lite)

---

## 📄 License

This project is for academic and educational purposes only.
