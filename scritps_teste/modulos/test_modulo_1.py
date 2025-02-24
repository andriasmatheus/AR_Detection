import sys
print(sys.executable)

import cv2
import time
from ultralytics import YOLO
import os
import openpifpaf

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
        window_width, window_height = 1280, 720
        cv2.namedWindow('YOLO11 Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO11 Detection', window_width, window_height)

        # Processar os frames do vídeo
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fim do vídeo ou erro ao ler o frame")
                break

            # Redimensionar o frame
            frame_resized = cv2.resize(frame, (window_width, window_height))

            # Fazer a detecção de objetos com YOLO, filtrando apenas pessoas (classe 0)
            results = model.predict(frame_resized, classes=[0])

            # Renderizar os resultados na imagem
            result_img = results[0].plot()

            # Exibir o frame com as detecções e poses
            cv2.imshow('YOLO11 Detection', result_img)

            # Pressione 'q' para sair do loop de exibição
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break

        # Liberar os recursos
        cap.release()
        cv2.destroyAllWindows()
