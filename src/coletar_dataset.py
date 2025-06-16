import cv2
import os
import time
import numpy as np
import mediapipe as mp

# Caminho onde as imagens serão salvas
SAVE_PATH = "../dataset"
os.makedirs(SAVE_PATH, exist_ok=True)

# Dimensões da imagem salva
image_width, image_height = 32, 32

# Controle da letra capturada e tempo de captura
capture_letter = None
save_delay = 0.2  # intervalo entre capturas (segundos)
last_save_time = 0
image_count = 0

# Inicializa MediaPipe para segmentação de fundo
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Inicializa MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Inicializa a webcam
cap = cv2.VideoCapture(0)
print(">>> Pressione a tecla correspondente à letra (a-z) para iniciar a coleta.")
print(">>> Pressione ESC para pausar. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a câmera.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_display = frame.copy()

    # Segmentação de fundo
    segmentation_results = selfie_segmentation.process(frame_rgb)
    condition = np.stack((segmentation_results.segmentation_mask,) * 3, axis=-1) > 0.5
    frame_black_bg = np.where(condition, frame_rgb, 0)

    # Detecção de mãos
    hand_detection = hands.process(frame_black_bg)

    if hand_detection.multi_hand_landmarks:
        for hand_landmarks in hand_detection.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
            y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
            x_min, x_max = max(0, min(x_coords)-20), min(w, max(x_coords)+20)
            y_min, y_max = max(0, min(y_coords)-20), min(h, max(y_coords)+20)

            # Recorta a região da mão
            hand_roi = frame_black_bg[y_min:y_max, x_min:x_max]

            # Exibe e salva a imagem
            if hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0:
                resized_hand = cv2.resize(hand_roi, (image_width, image_height))
                hand_display = cv2.cvtColor(resized_hand, cv2.COLOR_RGB2BGR)
                cv2.imshow("Recorte da Mão", hand_display)

                # Se uma letra estiver selecionada, salva a imagem com intervalo
                if capture_letter and (time.time() - last_save_time) > save_delay:
                    letter_dir = os.path.join(SAVE_PATH, capture_letter)
                    os.makedirs(letter_dir, exist_ok=True)
                    filename = f"{letter_dir}/{capture_letter}_{image_count:04d}.jpg"
                    cv2.imwrite(filename, hand_display)
                    print(f"[{capture_letter.upper()}] Imagem salva: {filename}")
                    image_count += 1
                    last_save_time = time.time()

    # Mostra a imagem principal
    cv2.imshow("Captura", frame_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):
        break
    elif ord('a') <= key <= ord('z'):
        capture_letter = chr(key)
        image_count = 0
        print(f">>> Iniciando coleta para a letra: {capture_letter.upper()}")
    elif key == 27:  # Tecla ESC
        capture_letter = None
        print(">>> Coleta pausada.")

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
hands.close()
selfie_segmentation.close()
