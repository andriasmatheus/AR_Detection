import sys
print(sys.executable)

import cv2
import time
from ultralytics import YOLO
import os
import openpifpaf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Carregar o modelo YOLO pré-treinado
model = YOLO("yolo11s.pt")
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30')

def detectar_pessoas(image_np):
    # Fazer a detecção de objetos com YOLO, filtrando apenas pessoas (classe 0)
    results = model.predict(image_np, classes=[0])
    pessoas_detectadas = results[0]

    return pessoas_detectadas

# Carregue sua imagem e converta para NumPy
image = Image.open('dataset/raw/Captura.png').convert('RGB')
image_np = np.array(image)

pessoas_detectadas = detectar_pessoas(image_np)

for box in pessoas_detectadas.boxes:
    # Extrair as coordenadas da caixa delimitadora
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    subimagem_np = image_np[y1:y2, x1:x2]

    fig, ax = plt.subplots(figsize=(10, 10))
    predictions, _, _ = predictor.numpy_image(subimagem_np)
    openpifpaf.show.AnnotationPainter().annotations(ax, predictions)
    ax.imshow(subimagem_np)
    plt.show()
