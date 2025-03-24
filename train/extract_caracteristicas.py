import cv2
import os
import numpy as np
from ultralytics import YOLO
import openpifpaf
import matplotlib.pyplot as plt
import math
import json

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def distancia_entre_articulacoes(ponto1, ponto2):
    # Desempacota as coordenadas de cada ponto
    x1, y1 = ponto1
    x2, y2 = ponto2
    
    # Calcula a distância utilizando a fórmula da distância euclidiana
    distancia = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distancia

def angulo_entre_articulacoes(vetor1, vetor2):
    # Desempacota as coordenadas dos vetores
    x1, y1 = vetor1
    x2, y2 = vetor2
    
    # Calcula o produto escalar
    produto_escalar = x1 * x2 + y1 * y2
    
    # Calcula a magnitude de cada vetor
    magnitude_vetor1 = math.sqrt(x1**2 + y1**2)
    magnitude_vetor2 = math.sqrt(x2**2 + y2**2)
    
    # Evita divisão por zero
    if magnitude_vetor1 == 0 or magnitude_vetor2 == 0:
        raise ValueError("A magnitude de um dos vetores é zero, o que torna o cálculo do ângulo inválido.")
    
    # Calcula o cosseno do ângulo
    cos_theta = produto_escalar / (magnitude_vetor1 * magnitude_vetor2)
    
    # Limita o valor de cos_theta para estar no intervalo [-1, 1]
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    # Calcula o ângulo em radianos
    angulo_radianos = math.acos(cos_theta)
    
    # Converte para graus, se necessário
    angulo_graus = math.degrees(angulo_radianos)
    
    return angulo_graus

def calcular_convex_hull(pontos):
    # Certifique-se de que 'pontos' é um numpy array
    pontos = np.array(pontos)

    # Calcular o Convex Hull
    hull = ConvexHull(pontos)
    """
    # Plotar os pontos e o Convex Hull
    plt.plot(pontos[:,0], pontos[:,1], 'o')  # Plotar pontos
    for simplex in hull.simplices:  # Plotar linhas do hull
        plt.plot(pontos[simplex, 0], pontos[simplex, 1], 'k-')
    plt.show()
    """
    
    return hull

def get_caracteristicas(prediction):
    caracteristicas = []

    keypoints = prediction.data.reshape(-1, 3)  # Reshape to (num_keypoints, 3) -> (x, y, confidence)
    keypoint_names = prediction.keypoints

    articulacoes = [[ponto[0], ponto[1]] for ponto in keypoints if ponto[2] > 0]

    convexHull = calcular_convex_hull(articulacoes)
    cHullArea = convexHull.area
    caracteristicas.append({"Área do Convex Hull": cHullArea})
    print("\n\nÁrea do Convex-Hull:", cHullArea)

    for i in range(len(articulacoes) - 1):
        distancia = distancia_entre_articulacoes(articulacoes[i], articulacoes[i + 1])
        angulo = angulo_entre_articulacoes(articulacoes[i], articulacoes[i + 1])

        textoDistancia = f"Distância Entre {keypoint_names[i]} e {keypoint_names[i+1]}: {distancia}"
        textoAngulo = f"Ângulo Entre {keypoint_names[i]} e {keypoint_names[i+1]}: {distancia}"

        caracteristicas.append({textoDistancia: distancia})
        print(textoDistancia)
        caracteristicas.append({textoAngulo: angulo})
        print(textoAngulo)

    print("-" * 50)
    return caracteristicas



def detectar_pessoas(image_np):
    # Fazer a detecção de objetos com YOLO, filtrando apenas pessoas (classe 0)
    results = model.predict(image_np, classes=[0])
    pessoas_detectadas = results[0]

    return pessoas_detectadas

train_path = os.path.abspath(r'dataset/raw/train') # Caminho para o diretório de vídeos

WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
model = YOLO("yolo11s.pt") # Carregar o modelo YOLO pré-treinado
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30') # Carregar o modelo OpenPifPaf para detecção de poses
# resnet50, shufflenetv2k30

stop = False
for root, dirs, files in os.walk(train_path):

    if stop: break    
    if root == train_path or "caracteristicas" in root: continue # Não olha os arquivos do diretório raiz

    for file in files:
        if stop: break
        if not file.endswith(".jpg"): continue
        video_path = os.path.join(root, file)

        capture = cv2.VideoCapture(video_path) # Abrir o vídeo com OpenCV

        if not capture.isOpened(): # Verifica se o vídeo foi aberto corretamente
            raise FileNotFoundError(f"Erro ao abrir o vídeo: {video_path}")

        while capture.isOpened(): # Processar cada frame do vídeo
            ret, frame = capture.read()
            if not ret:
                print("Fim do vídeo ou erro ao ler o frame")
                break

            # Redimensionar o frame
            frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            image_np = np.array(frame_resized)

            pessoas_detectadas = detectar_pessoas(image_np)

            # Obter coordenadas das detecções de pessoas (bounding boxes)
            boxes = pessoas_detectadas.boxes
            if len(boxes) == 0:
                print("Nenhuma pessoa detectada na cena!")

            caracteristicas_cena = []
            for box in boxes:
                # Extrair as coordenadas da caixa delimitadora
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                subimagem_np = image_np[y1:y2, x1:x2]

                predictions, _, _ = predictor.numpy_image(subimagem_np)

                # Verifique se há predições e imprima a estrutura para depuração
                if len(predictions) == 0:
                    print("Nenhuma articulação detectada no bounding box.")
                
                caracteristicas_box = []
                for prediction in predictions:
                    caracteristicas_box.append(get_caracteristicas(prediction))
                caracteristicas_cena.append(caracteristicas_box)
            with open(video_path.replace(".jpg", ".json").replace("imagens", "caracteristicas"), 'w', encoding='utf-8') as arquivo:
                json.dump(caracteristicas_cena, arquivo, ensure_ascii=False, indent=4)
        


