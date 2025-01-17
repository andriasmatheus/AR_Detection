import cv2
import time
from ultralytics import YOLO
import os
import openpifpaf

# Carregar o modelo YOLO pré-treinado
model = YOLO("yolo11s.pt")

# Inicializar o modelo OpenPifPaf
pifpaf_predictor = openpifpaf.Predictor()

# Caminho para o diretório de vídeos
train_path = r'dataset\raw\train'

stop = False
for root, dirs, files in os.walk(train_path):
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

            # Processar cada detecção para análise de pose
            for det in results[0].boxes:
                if det.cls == 0:  # Garantir que seja a classe "pessoa"
                    # Obter as coordenadas do bounding box
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    person_crop = frame_resized[y1:y2, x1:x2]

                    # Executar a detecção de pose com OpenPifPaf
                    predictions, _ = pifpaf_predictor.numpy_image(person_crop)

                    # Desenhar as poses detectadas no frame
                    for pred in predictions:
                        openpifpaf.show.annotation(frame_resized, pred)

            # Exibir o frame com as detecções e poses
            cv2.imshow('YOLO11 Detection', frame_resized)

            # Pressione 'q' para sair do loop de exibição
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break

        # Liberar os recursos
        cap.release()
        cv2.destroyAllWindows()
