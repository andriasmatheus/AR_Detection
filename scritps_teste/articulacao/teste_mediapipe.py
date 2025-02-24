import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Função para calcular a distância euclidiana entre dois pontos (x1, y1) e (x2, y2)
def euclidean_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Carregar uma imagem com OpenCV
image = cv2.imread('dataset/raw/Captura.png')

# Converter a imagem para RGB (OpenCV usa BGR por padrão)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Processar a imagem para detectar as poses
results = pose.process(image_rgb)

# Se a pose for detectada
if results.pose_landmarks:
    keypoints = results.pose_landmarks.landmark
    coordinates = []

    # Armazenar as coordenadas (x, y) das articulações
    for landmark in keypoints:
        coordinates.append((landmark.x, landmark.y))  # Posição normalizada entre 0 e 1 (relativa ao tamanho da imagem)

    # Calcular distâncias entre articulações específicas
    # Exemplo de distâncias que você pode querer calcular:
    # 1. Distância entre o ombro esquerdo e o quadril esquerdo
    left_shoulder = coordinates[11]  # 'left_shoulder' - índice 11
    left_hip = coordinates[23]      # 'left_hip' - índice 23
    distance_shoulder_hip = euclidean_distance(left_shoulder, left_hip)

    # 2. Distância entre o cotovelo esquerdo e o tornozelo esquerdo
    left_elbow = coordinates[13]     # 'left_elbow' - índice 13
    left_ankle = coordinates[25]    # 'left_ankle' - índice 25
    distance_elbow_ankle = euclidean_distance(left_elbow, left_ankle)

    # 3. Distância entre o pescoço (base da cabeça) e o tornozelo esquerdo
    neck = coordinates[11]           # 'left_shoulder' como exemplo de pescoço
    distance_neck_ankle = euclidean_distance(neck, left_ankle)

    # Mostrar as distâncias calculadas
    print(f'Distância entre ombro esquerdo e quadril esquerdo: {distance_shoulder_hip:.2f}')
    print(f'Distância entre cotovelo esquerdo e tornozelo esquerdo: {distance_elbow_ankle:.2f}')
    print(f'Distância entre pescoço e tornozelo esquerdo: {distance_neck_ankle:.2f}')

    # Desenhar as articulações na imagem
    for idx, (x, y) in enumerate(coordinates):
        if x != 0 and y != 0:  # Se a articulação foi detectada
            h, w, _ = image.shape
            x, y = int(x * w), int(y * h)  # Converter para coordenadas em pixels
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Desenha o ponto da articulação

    # Exibir a imagem com as articulações desenhadas
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Desativa os eixos
    output_path = 'imagem_processada_com_deteccoes.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.show()

else:
    print("Nenhuma pose detectada na imagem.")
