import numpy as np

# Exemplo de dados
n_samples = 1  # Número de vídeos ou sequências
n_frames = 24    # Número de frames por sequência de vídeo
width = 1920       # Largura da imagem
height = 1080      # Altura da imagem
channels = 3     # Número de canais (1 para imagem em escala de cinza, 3 para RGB)

# Gerando dados aleatórios para simular os frames (normalmente, você carregaria seus vídeos aqui)
X_train = np.random.randn(n_samples, n_frames, height, width, channels)  # Forma: (100, 10, 64, 64, 1)

# Gerando rótulos (labels) aleatórios (se for classificação binária, por exemplo)
y_train = np.random.randint(0, 2, n_samples)  # Labels binários para este exemplo

print("Forma dos dados de entrada (X_train):", X_train.shape)
print("Forma dos rótulos (y_train):", y_train.shape)