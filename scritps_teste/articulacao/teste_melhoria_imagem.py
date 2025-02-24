import cv2
import mediapipe as mp

# Carregar a imagem
image = cv2.imread('dataset/raw/Captura.png')

# Ajustar brilho e contraste
alpha = 1.5  # Fator de contraste
beta = 20    # Fator de brilho
adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Aumentar resolução da imagem
image_resized = cv2.resize(adjusted_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Converter para RGB (MediaPipe requer imagens RGB)
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# Processar a imagem para detectar as articulações
results = pose.process(image_rgb)

# Visualizar os resultados
if results.pose_landmarks:
    for landmark in results.pose_landmarks.landmark:
        print(f"Landmark: {landmark}")
        # Desenhar as articulações na imagem
        x = int(landmark.x * image_resized.shape[1])
        y = int(landmark.y * image_resized.shape[0])
        cv2.circle(image_resized, (x, y), 5, (0, 0, 255), -1)

# Exibir a imagem com as articulações
cv2.imshow("Pose Detection", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
