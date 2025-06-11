import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp

model_path = 'asl_gesture_model2.h5'
model = tf.keras.models.load_model(model_path)
print("Model sucessfully loaded.")

image_width, image_height = 32, 32
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error. Camera not accessible.")
    exit()

# Segmentação de fundo (imagens foram treinadas com fundo preto, aí é preciso segmentar o fundo para que o modelo 
# funcione corretamente e destaque a mão)
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Detecção de mãos no mediapipe (para detectar a mão e fazer o crop da região da mão)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

print("Processing frames...")
print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error. Frame not acessible.")
        break

    # Converte o frame para RGB e cria uma cópia para exibição
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_display = frame.copy()

    # Variáveis para armazenar o resultado da predição (confiança e letra prevista)
    predicted_letter = "N/A"
    confidence = 0.0

    # Processamento de segmentação de fundo e detecção de mãos
    segmentation_results = selfie_segmentation.process(frame_rgb) # Segmentação do fundo
    condition = np.stack((segmentation_results.segmentation_mask,) * 3, axis=-1) > 0.5 # Cria uma máscara para o fundo
    frame_with_black_bg_rgb = np.where(condition, frame_rgb, 0) # Aplica a máscara para deixar o fundo preto

    hand_detection_results = hands.process(frame_with_black_bg_rgb) # Detecção de mãos a partir do frame com fundo preto

    # Se mãos forem detectadas, desenha as landmarks e faz o crop da região da mão
    if hand_detection_results.multi_hand_landmarks:
        for hand_landmarks in hand_detection_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Desenha as landmarks da mão no frame de exibição

            h, w, c = frame_rgb.shape
            x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            hand_roi_rgb = frame_with_black_bg_rgb[y_min:y_max, x_min:x_max]

            # Verifica se a região da mão é válida antes de processar
            if hand_roi_rgb.shape[0] > 0 and hand_roi_rgb.shape[1] > 0:
                resized_hand = cv2.resize(hand_roi_rgb, (image_width, image_height))
                normalized_hand = resized_hand.astype(np.float32) / 255.0
                input_model = np.expand_dims(normalized_hand, axis=0)

                predictions = model.predict(input_model, verbose=0) # Pega o frame da mão processada e faz a predição/frame
                predicted_class_index = np.argmax(predictions[0])
                predicted_letter = class_names[predicted_class_index]
                confidence = predictions[0][predicted_class_index]
            else:
                predicted_letter = "No valid hand ROI"
    else:
        predicted_letter = "No hand detected"

    display_text = f"Pred: {predicted_letter} ({confidence:.2f})"
    cv2.putText(frame_display, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('ASL Gesture Recognition (Live)', frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
selfie_segmentation.close()