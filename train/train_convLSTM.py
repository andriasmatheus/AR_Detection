import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

K.clear_session()

# Função para carregar frames de vídeo
def load_video_frames(video_path, target_height, target_width, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)
        
        if len(frames) >= max_frames:
            break
    
    cap.release()
    
    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=-1)
    
    return frames

# Função para carregar vídeos da pasta
def load_videos_from_folder(folder_path, target_height, target_width, max_frames=100):
    subfolders = ['passagem_segura', 'passagem_insegura', 'acao_autorizada', 'acao_nao_autorizada']
    
    X = []
    y = []
    
    label_map = {'passagem_segura': 0, 'passagem_insegura': 1, 'acao_autorizada': 2, 'acao_nao_autorizada': 3}
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            video_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
            
            for video_file in video_files:
                video_path = os.path.join(subfolder_path, video_file)
                
                # Carrega os frames
                frames = load_video_frames(video_path, target_height, target_width, max_frames)

                # Gerar vetores de características (exemplo simples, substituir pela extração real)
                caracteristicas = np.random.rand(frames.shape[0], 10)  # Exemplo: 10 características por frame
                
                # Concatenar características com os frames
                frames_com_caracteristicas = np.concatenate([frames, np.expand_dims(caracteristicas, axis=-1)], axis=-1)
                
                label = label_map[subfolder]
                X.append(frames_com_caracteristicas)
                y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Definindo parâmetros
video_folder = 'dataset/raw/train'
target_height = 60
target_width = 60

# Carregar vídeos e rótulos
X_train, y_train = load_videos_from_folder(video_folder, target_height, target_width)

# Normalização
X_train = X_train.astype('float32') / 255.0

# Construir o modelo ConvLSTM
model = Sequential()

model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False))
model.add(BatchNormalization())
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
