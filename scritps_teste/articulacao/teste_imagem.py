import cv2
import os
import numpy as np
from ultralytics import YOLO
import openpifpaf
import matplotlib.pyplot as plt
import tempfile

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def distancia_entre_articulacoes(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def angulo_entre_articulacoes(p1, p2, p3):
    # Vetores a partir de p2
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Cálculo do ângulo
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Corrigir para precisão numérica
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Retornar o ângulo em graus
    return np.degrees(np.arccos(cos_theta))

def calcular_convex_hull(pontos):
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

# Carregar o modelo YOLO pré-treinado
model = YOLO("yolo11s.pt")

# Carregar o modelo OpenPifPaf para detecção de poses (use mobilenetv2)
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30')


from PIL import Image
# Carregue sua imagem e converta para NumPy
image = Image.open('dataset/raw/capture.png')
image_np = np.array(image)

# Definir a largura e altura da janela de exibição
window_width, window_height = 1280, 720
cv2.namedWindow('YOLO11 Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO11 Detection', window_width, window_height)

# Redimensionar o frame
frame_resized = cv2.resize(image_np, (window_width, window_height))

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

        # Verifique se há predições e imprima a estrutura para depuração
        if len(predictions) > 0:
            print("Predições de articulações:", predictions)
            
            # Acesse as articulações de forma segura
            articulacoes = predictions[0].skeleton
            
            # Calcule as distâncias e ângulos
            distancia = distancia_entre_articulacoes(articulacoes[0], articulacoes[1])
            print(f'Distância entre o ombro e o cotovelo: {distancia}')

            angulo = angulo_entre_articulacoes(articulacoes[0], articulacoes[1], articulacoes[2])
            print(f'Ângulo entre o ombro, cotovelo e pulso: {angulo} graus')

            # Calcular o Convex Hull para todas as articulações
            convex_hull = calcular_convex_hull(articulacoes)
        else:
            print("Nenhuma articulação detectada no quadro.")
# Exibir a imagem final com as detecções e poses
cv2.imshow('YOLO11 Detection', frame_resized)

# Esperar até que uma tecla seja pressionada para fechar a janela
cv2.waitKey(0)
if cv2.waitKey(1) & 0xFF == ord('q'):
    # Fechar todas as janelas do OpenCV
    cv2.destroyAllWindows()