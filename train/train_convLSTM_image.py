import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from PIL import Image
import json

K.clear_session()

def extrair_caracteristicas_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as arquivo:
        caracteristicas = json.load(arquivo)

    valores = []

    # Função recursiva para lidar com estruturas aninhadas
    def extrair_recursivo(dado):
        if isinstance(dado, list):
            for item in dado:
                extrair_recursivo(item)
        elif isinstance(dado, dict):
            for chave, valor in dado.items():
                valores.append(valor)  # Adiciona o valor no vetor de valores
                extrair_recursivo(valor)  # Chama recursivamente caso o valor seja um objeto ou lista

    # Iniciar o processo de extração
    extrair_recursivo(caracteristicas)

    return valores

# Função para carregar vídeos da pasta
def load_videos_from_folder(folder_path, target_height, target_width, max_frames=100):
    subfolders = ['passagem_segura', 'passagem_insegura']
                  
    X = []
    y = []
    
    label_map = {'passagem_segura': 0, 'passagem_insegura': 1}
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        
        if not os.path.isdir(subfolder_path): continue
        
        list_frames = [f for f in os.listdir(subfolder_path) if f.endswith(('.jpg'))]
        
        # Organizar os frames em grupos (sequências)
        frames = []
        
        for frame_path in list_frames:
            path = os.path.join(subfolder_path, frame_path)
            
            frame = Image.open(path).convert('RGB')
            frame = frame.resize((target_width, target_height))
            frame_np = np.array(frame)

            #print(frame_np)

            caracteristicas = extrair_caracteristicas_json(path.replace("imagens", "caracteristicas").replace(".jpg", ".json"))

            num_linhas = frame_np.shape[0]
            if len(caracteristicas) < num_linhas:
                caracteristicas.extend([0] * (num_linhas - len(caracteristicas)))  # Preenche com 0 caso haja menos dados

            valores_json = np.array(caracteristicas).reshape(-1, 1)  # Transformando para (60, 1)
            valores_json = np.tile(valores_json, (60, 1, 1))  # Replicando para (60, 60, 1)

            # Agora usamos np.concatenate para adicionar esses valores como uma nova dimensão (no eixo 2)
            frame_with_features = np.concatenate((frame_np, valores_json), axis=-1)
            # print(frame_with_features)
            frames.append(frame_with_features)
            
            # Quando atingimos o número máximo de frames, tratamos como um vídeo
            if len(frames) == max_frames:
                X.append(np.array(frames))
                y.append(label_map[subfolder])
                frames = []  # Limpar a lista para a próxima sequência
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalizar os dados (se necessário)
    X = X / 255.0  # Opcional, se necessário normalizar
    
    return X, y

# Definindo parâmetros
video_folder = 'dataset/raw/train/imagens'
target_height = 60
target_width = 60
# Carregar vídeos e rótulos
X_train, y_train = load_videos_from_folder(video_folder, target_height, target_width)

print(X_train)
print(y_train)

# Construir o modelo ConvLSTM
model = Sequential()

model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='sa                 me', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], 4)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout para evitar overfitting
model.add(Dense(4, activation='softmax'))

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Treinar o modelo
history = model.fit(X_train, y_train, batch_size=2, epochs=10, validation_split=0.2)

# Salvar o modelo
model.save('conv_lstm_model.h5')

# Carregar o modelo
# model = tf.keras.models.load_model('conv_lstm_model.h5')