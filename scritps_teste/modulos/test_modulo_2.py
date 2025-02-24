import cv2
import os
import numpy as np
from ultralytics import YOLO
import openpifpaf
import matplotlib.pyplot as plt
import tempfile

# Carregar o modelo YOLO pré-treinado
model = YOLO("yolo11s.pt")

# Carregar o modelo OpenPifPaf para detecção de poses (use mobilenetv2)
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30')

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

                    # Realizar a detecção de poses para a imagem da pessoa
                    person_image_np = np.array(person_image_resized)
                    predictions, _, _ = predictor.numpy_image(person_image_np)

                    # Visualizar a detecção de poses sobre a pessoa
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(person_image_resized)
                    openpifpaf.show.AnnotationPainter().annotations(ax, predictions)
                    plt.axis('off')

                    # Salvar a visualização em um arquivo temporário
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        temp_file_path = temp_file.name
                        plt.savefig(temp_file_path, bbox_inches='tight', pad_inches=0)
                        plt.close(fig)

                    # Carregar a imagem salva e convertê-la para um formato compatível com OpenCV
                    pose_image = cv2.imread(temp_file_path)

                    # Redimensionar a imagem da pose para o tamanho da caixa delimitadora original
                    pose_image_resized = cv2.resize(pose_image, (x2 - x1, y2 - y1))

                    # Substitui a área do frame com a detecção de pose sobre a pessoa
                    frame_resized[y1:y2, x1:x2] = pose_image_resized

                    # Remover o arquivo temporário após o uso
                    os.remove(temp_file_path)

            # Exibir o frame com as detecções e poses
            cv2.imshow('YOLO11 Detection', frame_resized)

            # Pressione 'q' para sair do loop de exibição
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break

        # Liberar os recursos
        cap.release()
        cv2.destroyAllWindows()
