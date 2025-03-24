import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Carregar o modelo DeepLabV3 pré-treinado
model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)

# Função para realizar a segmentação semântica
def semantic_segmentation(image_path):
    # Carregar a imagem
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converter BGR para RGB

    # Redimensionar a imagem para o tamanho esperado pelo modelo
    image_resized = cv2.resize(image, (224, 224))
    image_resized = np.expand_dims(image_resized, axis=0)
    image_resized = tf.keras.applications.densenet.preprocess_input(image_resized)

    # Realizar a predição
    predictions = model.predict(image_resized)

    # A saída do modelo será um mapa de características com várias dimensões
    # Aqui, queremos reduzir isso a uma forma de imagem
    prediction_map = np.argmax(predictions[0], axis=-1)  # Pegue a classe com maior probabilidade
    
    # Redimensione o mapa de volta ao tamanho original da imagem
    prediction_map_resized = cv2.resize(prediction_map.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Mostrar a imagem original e a segmentação
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(prediction_map_resized, cmap='jet')  # Usando o mapa de cores 'jet' para a segmentação
    plt.title('Segmentação Semântica')
    plt.axis('off')

    plt.show()

# Caminho da imagem a ser segmentada
image_path = 'dataset/raw/frame0_objeto3.jpg'  # Substitua pelo caminho da sua imagem
semantic_segmentation(image_path)
