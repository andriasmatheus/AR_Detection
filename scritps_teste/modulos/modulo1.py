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

def detectar_pessoas(image_np):
    # Fazer a detecção de objetos com YOLO, filtrando apenas pessoas (classe 0)
    results = model.predict(image_np, classes=[0])
    pessoas_detectadas = results[0]

    return pessoas_detectadas

# Carregue sua imagem e converta para NumPy
image = Image.open('dataset/raw/Captura.png').convert('RGB')
image_np = np.array(image)

pessoas_detectadas = detectar_pessoas(image_np)

# Renderizar os resultados na imagem
result_img = pessoas_detectadas.plot()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(result_img)
plt.axis('off')

output_path = 'imagem_processada_com_deteccoes.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.show()