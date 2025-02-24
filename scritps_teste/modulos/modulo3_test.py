import cv2
import os
import numpy as np
from ultralytics import YOLO
import openpifpaf
import matplotlib.pyplot as plt
import tempfile

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

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
    
    # Evita divisão por zero
    if magnitude_vetor1 == 0 or magnitude_vetor2 == 0:
        raise ValueError("A magnitude de um dos vetores é zero, o que torna o cálculo do ângulo inválido.")
    
    # Calcula o cosseno do ângulo
    cos_theta = produto_escalar / (magnitude_vetor1 * magnitude_vetor2)
    
    # Limita o valor de cos_theta para estar no intervalo [-1, 1]
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    # Calcula o ângulo em radianos
    angulo_radianos = math.acos(cos_theta)
    
    # Converte para graus, se necessário
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

def get_caracteristicas(prediction):
    caracteristicas = []

    keypoints = prediction.data.reshape(-1, 3)  # Reshape to (num_keypoints, 3) -> (x, y, confidence)
    keypoint_names = prediction.keypoints

    articulacoes = [[ponto[0], ponto[1]] for ponto in keypoints if ponto[2] > 0]

    convexHull = calcular_convex_hull(articulacoes)
    cHullArea = convexHull.area
    caracteristicas.append({"Área do Convex Hull": cHullArea})
    print("\n\nÁrea do Convex-Hull:", cHullArea)

    for i in range(len(articulacoes) - 1):
        distancia = distancia_entre_articulacoes(articulacoes[i], articulacoes[i + 1])
        angulo = angulo_entre_articulacoes(articulacoes[i], articulacoes[i + 1])

        textoDistancia = f"Distância Entre {keypoint_names[i]} e {keypoint_names[i+1]}: {distancia}"
        textoAngulo = f"Ângulo Entre {keypoint_names[i]} e {keypoint_names[i+1]}: {distancia}"

        caracteristicas.append({textoDistancia: distancia})
        print(textoDistancia)
        caracteristicas.append({textoAngulo: angulo})
        print(textoAngulo)

    print("-" * 50)
    return caracteristicas

train_path = os.path.abspath(r'dataset/raw/train') # Caminho para o diretório de vídeos

model = YOLO("yolo11s.pt") # Carregar o modelo YOLO pré-treinado
predictor = openpifpaf.Predictor(checkpoint='resnet50') # Carregar o modelo OpenPifPaf para detecção de poses
# resnet50, shufflenetv2k30

stop = False
for root, dirs, files in os.walk(train_path):
    print("Entrou")
    if stop:
        break

    for file in files:
        if stop:
            break
        video_path = os.path.join(root, file)

        # Abrir o vídeo pré-gravado com OpenCV
        cap = cv2.VideoCapture(video_path)

        # Verificar se o vídeo foi aberto corretamente
        if not cap.isOpened():
            raise FileNotFoundError(f"Erro ao abrir o vídeo: {video_path}")

        # Definir a largura e altura da janela de exibição
        window_width, window_height = 1280, 720
        cv2.namedWindow('YOLO11 Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO11 Detection', window_width, window_height)

        # Processar os frames do vídeo
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fim do vídeo ou erro ao ler o frame")
                break

            # Redimensionar o frame
            frame_resized = cv2.resize(frame, (window_width, window_height))

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
                        for i, prediction in enumerate(predictions):
                            keypoints = prediction.data.reshape(-1, 3)
                            keypoint_names = prediction.keypoints
                            """
                            print(f"Person {i+1}:")
                            for j, (x, y, confidence) in enumerate(keypoints):
                                if (confidence > 0):
                                    print(f"  Keypoint {keypoint_names[j]}: (x={x:.2f}, y={y:.2f}, confidence={confidence:.2f})")
                            """
                            get_caracteristicas(prediction)
                    else:
                        print("Nenhuma articulação detectada no quadro.")

            # Exibir o frame com as detecções e poses
            cv2.imshow('YOLO11 Detection', frame_resized)

            # Pressione 'q' para sair do loop de exibição
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break

        # Liberar os recursos
        cap.release()
        cv2.destroyAllWindows()
