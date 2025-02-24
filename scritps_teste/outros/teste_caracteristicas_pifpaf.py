import numpy as np
import openpifpaf
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math

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
    
    # Calcula o cosseno do ângulo
    cos_theta = produto_escalar / (magnitude_vetor1 * magnitude_vetor2)
    
    # Calcula o ângulo em radianos
    angulo_radianos = math.acos(cos_theta)
    
    # Converte o ângulo para graus
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

# Carregue sua imagem e converta para NumPy
image = Image.open('dataset/raw/Captura.png').convert('RGB')
image_np = np.array(image)

# Processamento de imagem com OpenPifPaf
predictor = openpifpaf.Predictor(checkpoint='resnet50')
predictions, _, _ = predictor.numpy_image(image_np)

# Display detected keypoints
for i, pred in enumerate(predictions):
    keypoints = pred.data.reshape(-1, 3)  # Reshape to (num_keypoints, 3) -> (x, y, confidence)
    keypoint_names = pred.keypoints

    print(f"Person {i+1}:")
    for j, (x, y, confidence) in enumerate(keypoints):
        if (confidence > 0):
            print(f"  Keypoint {keypoint_names[j]}: (x={x:.2f}, y={y:.2f}, confidence={confidence:.2f})")
    
    articulacoes = [[ponto[0], ponto[1]] for ponto in keypoints if ponto[2] > 0]

    convexHull = calcular_convex_hull(articulacoes)
    cHullArea = convexHull.area
    print("\n\nÁrea do Convex-Hull:", cHullArea)

    distancia = distancia_entre_articulacoes(articulacoes[0], articulacoes[1])
    print("Distância Entre {pontoA}, {pontoB}:", distancia)

    angulo = angulo_entre_articulacoes(articulacoes[0], articulacoes[1])
    print("Distância Entre {pontoA}, {pontoB}:", angulo)

    print("-" * 50)

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
