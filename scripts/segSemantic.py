import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para detectar bordas usando o Canny
def detect_edges(image_path, low_threshold=800, high_threshold=150):
    # Carregar a imagem
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converte a imagem para escala de cinza

    # Aplicar o Canny para detectar as bordas
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)

    # Mostrar a imagem original e a imagem com as bordas detectadas
    plt.figure(figsize=(10, 5))

    # Exibe a imagem original
    plt.subplot(1, 2, 1)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(original_image)
    plt.title("Imagem Original")

    # Exibe as bordas detectadas
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Bordas Detectadas (Canny)")

    plt.show()


def rgb_to_hsv_histogram(image_path):
    # Carregar a imagem em RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Converter a imagem para HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Separar os canais
    h, s, v = cv2.split(hsv_image)
    
    # Aplicar um filtro Gaussiano para suavizar a imagem
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordas usando Canny com parâmetros ajustados
    edges = cv2.Canny(blurred, 50, 150)
    
    # Criar uma máscara de bordas brancas sobre a imagem HSV
    edges_colored = cv2.merge([edges, edges, edges])  # Converter para 3 canais
    hsv_edges = cv2.addWeighted(hsv_image, 1, edges_colored, 1, 0)
    
    # Mostrar imagem HSV com bordas sobrepostas
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(hsv_image)
    plt.title('Imagem em HSV')
    plt.axis('off')
    
    # Criar histogramas
    plt.subplot(1, 3, 2)
    plt.hist(h.ravel(), bins=256, range=[0, 256], color='red', alpha=0.7, label='Hue')
    plt.hist(s.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7, label='Saturation')
    plt.hist(v.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7, label='Value')
    plt.title('Histogramas dos Canais HSV')
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    plt.legend()
    
    # Mostrar imagem com detecção de bordas sobreposta
    plt.subplot(1, 3, 3)
    plt.imshow(hsv_edges)
    plt.title('Imagem HSV com Detecção de Bordas')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()



# Caminho para a imagem que você quer processar
image_path = '../images/5_tr21.png'

# Detectar e marcar as bordas
#detect_edges(image_path)
rgb_to_hsv_histogram(image_path)


