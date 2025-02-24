import numpy as np
import openpifpaf
from PIL import Image
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull

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

for prediction in predictions:
    articulacoes = prediction.skeleton

    # Calcular o Convex Hull para todas as articulações
    convex_hull = calcular_convex_hull(articulacoes)
    print("Convex-hull Calculado")

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