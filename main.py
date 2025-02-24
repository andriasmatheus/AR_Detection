"""
import cv2
import time
from ultralytics import YOLO

# Carregar o modelo YOLO pré-treinado
model = YOLO("yolo11s.pt")

# Abrir o vídeo pré-gravado com OpenCV
video_path = r'dataset\raw\train\0_safe_walkway_violation\0_tr13.mp4'
cap = cv2.VideoCapture(video_path)

# Verificar se o vídeo foi aberto corretamente
if not cap.isOpened():
    raise FileNotFoundError(f"Erro ao abrir o vídeo: {video_path}")

# Definir a largura e altura da janela de exibição
window_width, window_height = 1280, 720
cv2.namedWindow('YOLOv11 Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLOv11 Detection', window_width, window_height)

# Configurar o VideoWriter para salvar o vídeo de saída
output_video_path = 'final.avi'
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Garantir um valor padrão para FPS
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (window_width, window_height))

# Processar os frames do vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro ao ler o frame")
        break

    # Redimensionar o frame
    frame_resized = cv2.resize(frame, (window_width, window_height))

    # Fazer a detecção de objetos com YOLO
    results = model.predict(frame_resized)

    # Renderizar os resultados na imagem
    result_img = results[0].plot()

    # Exibir o frame com as detecções
    cv2.imshow('YOLOv5 Detection', result_img)

    # Salvar o frame no vídeo de saída
    out.write(result_img)

    # Pressione 'q' para sair do loop de exibição
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
out.release()
cv2.destroyAllWindows()
"""

import numpy as np
import openpifpaf
from PIL import Image
import matplotlib.pyplot as plt

# Carregue sua imagem e converta para NumPy
image = Image.open('dataset/raw/capture.png').convert('RGB')
image_np = np.array(image)

# Processamento de imagem com OpenPifPaf
predictor = openpifpaf.Predictor(checkpoint='resnet50')
predictions, _, _ = predictor.numpy_image(image_np)

# Visualização
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image_np)
openpifpaf.show.AnnotationPainter().annotations(ax, predictions)
plt.axis('off')

# Salvar a imagem final
output_path = 'imagem_processada_com_deteccoes.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Imagem salva em: {output_path}")
