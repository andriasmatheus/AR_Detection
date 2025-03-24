import cv2
import os
import numpy as np

def load_video_frames(video_path, target_height, target_width):
    """
    Carrega todos os frames de um vídeo e os redimensiona para o formato esperado.
    Retorna uma sequência de frames no formato (n_frames, height, width, channels).
    """
    # Abre o vídeo com OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        # Lê o próximo frame
        ret, frame = cap.read()
        
        # Se não há mais frames, sai do loop
        if not ret:
            break
        
        # Converte para escala de cinza (ou RGB, dependendo do seu modelo)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Para escala de cinza
        # Redimensiona para o tamanho desejado
        frame = cv2.resize(frame, (target_width, target_height))
        
        # Adiciona o frame à lista
        frames.append(frame)
    
    cap.release()
    
    # Converte para um array numpy e adiciona a dimensão de canais
    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=-1)  # Adiciona a dimensão do canal (ex: 1 para cinza)
    
    return frames

def load_videos_from_folder(folder_path, target_height, target_width):
    """
    Carrega todos os vídeos de uma pasta e extrai os frames, atribuindo um rótulo baseado na subpasta.
    """
    # Lista as subpastas dentro do diretório principal
    subfolders = ['passagem_segura', 'passagem_insegura', 'acao_autorizada', 'acao_nao_autorizada']
    
    X = []
    y = []  # Aqui vamos armazenar os rótulos dos vídeos
    
    # Mapeamento de subpastas para rótulos
    label_map = {
        'passagem_segura': 0,
        'passagem_insegura': 1,
        'acao_autorizada': 2,
        'acao_nao_autorizada': 3
    }

    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        
        # Verifica se a subpasta existe
        if os.path.isdir(subfolder_path):
            video_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
            
            for video_file in video_files:
                video_path = os.path.join(subfolder_path, video_file)
                
                # Carrega os frames do vídeo
                frames = load_video_frames(video_path, target_height, target_width)
                
                # Atribui o rótulo baseado na subpasta
                label = label_map[subfolder]
                
                # Adiciona os frames e o rótulo
                X.append(frames)
                y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Parâmetros
video_folder = 'dataset/raw/train'  # Caminho para a pasta contendo os vídeos
target_height = 1080  # Altura das imagens
target_width = 1920  # Largura das imagens

# Carregar os vídeos e seus rótulos
X_train, y_train = load_videos_from_folder(video_folder, target_height, target_width)

# Verificar as formas dos dados carregados
print("Forma dos dados de entrada (X_train):", X_train.shape)  # Esperado: (n_samples, n_frames, height, width, channels)
print("Forma dos rótulos (y_train):", y_train.shape)
