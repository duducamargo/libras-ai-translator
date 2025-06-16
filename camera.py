import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
import json

# Carrega o modelo
model_path = 'asl_gesture_model2.h5'
model = tf.keras.models.load_model(model_path)
print("✅ Model successfully loaded.")

# Carrega o mapeamento de classes
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverte o mapeamento: {0: 'a', 1: 'b', ...}
index_to_class = {v: k for k, v in class_indices.items()}

image_width, image_height = 32, 32

# Inicializa webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Camera not accessible.")
    exit()

# Segmentação de fundo com MediaPipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Detecção de mãos com MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

print("🎥 Processing frames... (press 'q' to exit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Frame not accessible.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_display = frame.copy()

    predicted_letter = "N/A"
    confidence = 0.0

    # Segmentação do fundo
    segmentation_results = selfie_segmentation.process(frame_rgb)
    condition = np.stack((segmentation_results.segmentation_mask,) * 3, axis=-1) > 0.5
    frame_with_black_bg_rgb = np.where(condition, frame_rgb, 0)

    # Detecção da mão
    hand_detection_results = hands.process(frame_with_black_bg_rgb)

    if hand_detection_results.multi_hand_landmarks:
        for hand_landmarks in hand_detection_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
            y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]

            x_min = max(0, min(x_coords) - 20)
            y_min = max(0, min(y_coords) - 20)
            x_max = min(w, max(x_coords) + 20)
            y_max = min(h, max(y_coords) + 20)

            cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            hand_roi_rgb = frame_with_black_bg_rgb[y_min:y_max, x_min:x_max]

            if hand_roi_rgb.shape[0] > 0 and hand_roi_rgb.shape[1] > 0:
                resized_hand = cv2.resize(hand_roi_rgb, (image_width, image_height))
                normalized_hand = resized_hand.astype(np.float32) / 255.0
                input_model = np.expand_dims(normalized_hand, axis=0)

                predictions = model.predict(input_model, verbose=0)
                predicted_class_index = int(np.argmax(predictions[0]))
                predicted_letter = index_to_class.get(predicted_class_index, "Unknown")
                confidence = predictions[0][predicted_class_index]
            else:
                predicted_letter = "No valid hand ROI"
    else:
        predicted_letter = "No hand detected"

    display_text = f"Pred: {predicted_letter} ({confidence:.2f})"
    cv2.putText(frame_display, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if confidence >= 0.75 else (0, 0, 255), 2)

    cv2.imshow('ASL Gesture Recognition (Live)', frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
hands.close()
selfie_segmentation.close()
