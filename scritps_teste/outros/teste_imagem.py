import cv2
import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import matplotlib.pyplot as plt
import tempfile

# Função para ajustar brilho e contraste
def adjust_brightness_contrast(image, alpha=1.5, beta=20):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Inicializar o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.6)

# Carregar o modelo YOLO pré-treinado
model = YOLO("yolo11s.pt")

# Caminho para o diretório de vídeos
train_path = os.path.abspath(r'dataset/raw/train')
print("OK")
stop = False
for root, dirs, files in os.walk(train_path):
    print("Entrou")
    if stop:
        break

    for file in files:
        if stop:
            break
        video_path = os.path.join(root, file)

        # Abrir o vídeo pré-gravado com OpenCV
        cap = cv2.VideoCapture(video_path)

        # Verificar se o vídeo foi aberto corretamente
        if not cap.isOpened():
            raise FileNotFoundError(f"Erro ao abrir o vídeo: {video_path}")

        # Definir a largura e altura da janela de exibição
        window_width, window_height = 640, 360
        cv2.namedWindow('YOLO11 Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO11 Detection', window_width, window_height)

        # Processar os frames do vídeo
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fim do vídeo ou erro ao ler o frame")
                break

            frame_resized = cv2.resize(frame, (window_width, window_height))

            frame_resized = cv2.fastNlMeansDenoisingColored(frame_resized, None, 5, 7, 21)
            frame_resized = adjust_brightness_contrast(frame_resized)

            # Redimensionar o frame

            # Fazer a detecção de objetos com YOLO, filtrando apenas pessoas (classe 0)
            results = model.predict(frame_resized, classes=[0])

            # Obter coordenadas das detecções de pessoas (bounding boxes)
            boxes = results[0].boxes
            if len(boxes) > 0:
                for box in boxes:
                    # Extrair as coordenadas da caixa delimitadora
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Desenhar a bounding box ao redor da pessoa detectada
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Cortar a imagem da pessoa detectada
                    person_image = frame_resized[y1:y2, x1:x2]

                    # Aumentar a resolução da pessoa detectada (exemplo: aumentar 1.5x)
                    person_image_resized = cv2.resize(person_image, (person_image.shape[1] * 2, person_image.shape[0] * 2))

                    # Converter a imagem para RGB para o MediaPipe
                    person_image_rgb = cv2.cvtColor(person_image_resized, cv2.COLOR_BGR2RGB)

                    # Processar a imagem com o MediaPipe Pose
                    results_pose = pose.process(person_image_rgb)

                    # Verificar se a pose foi detectada
                    if results_pose.pose_landmarks:
                        landmarks = results_pose.pose_landmarks.landmark

                        # Desenhar as articulações detectadas (landmarks)
                        for landmark in landmarks:
                            # Converter coordenadas normalizadas para pixels
                            x = int(landmark.x * person_image_resized.shape[1])
                            y = int(landmark.y * person_image_resized.shape[0])
                            cv2.circle(person_image_resized, (x, y), 5, (0, 0, 255), -1)

                    # Ajustar as dimensões da imagem redimensionada da pessoa para se ajustar à área da caixa delimitadora
                    person_image_resized = cv2.resize(person_image_resized, (x2 - x1, y2 - y1))

                    # Substituir a imagem da pessoa no frame com as articulações detectadas
                    frame_resized[y1:y2, x1:x2] = person_image_resized

            # Exibir o frame com as detecções e poses
            cv2.imshow('YOLO11 Detection', frame_resized)

            # Pressione 'q' para sair do loop de exibição
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break

        # Liberar os recursos
        cap.release()
        cv2.destroyAllWindows()
