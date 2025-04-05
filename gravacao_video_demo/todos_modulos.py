import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random

model_path_yolo = "runs/detect/train10/weights/best.pt"

model = YOLO(model_path_yolo)

# Cores fixas para cada classe (pode ser global)
cores_classes = {}

def cor_para_classe(nome_classe):
    if nome_classe not in cores_classes:
        # Gerar uma cor RGB aleatória
        cores_classes[nome_classe] = tuple(random.randint(0, 255) for _ in range(3))
    return cores_classes[nome_classe]

def detecta_objetos(frame_do_video):
   
    # Realizar a detecção no frame
    results = model(frame_do_video)

    # 'results' é uma lista de objetos de detecção, precisamos acessar o primeiro item (que é o resultado da detecção no frame)
    result = results[0]

    objetos_detectados = []
    for box in result.boxes:
        class_id = int(box.cls[0])  # Índice da classe
        class_name = result.names[class_id]  # Nome da classe
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da caixa

        objeto = {
            "classe": class_name,
            "coordenadas": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        }
        objetos_detectados.append(objeto)
    return objetos_detectados

def desenha_retangulos_objetos(coordenadas_objetos, frame):
    for coordenada in coordenadas_objetos:
        pos = coordenada["coordenadas"]
        classe = coordenada["classe"]

        cor = cor_para_classe(classe)

        # Desenhar retângulo da detecção
        cv2.rectangle(frame, (pos["x1"], pos["y1"]), (pos["x2"], pos["y2"]), cor, 2)

        # Escrever o nome da classe
        texto = classe
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        escala_fonte = 0.8
        espessura = 2

        # Medir tamanho do texto
        (largura_texto, altura_texto), _ = cv2.getTextSize(texto, fonte, escala_fonte, espessura)

        # Posição padrão: texto acima da caixa
        y_texto = pos["y1"] - 5
        y_fundo = pos["y1"] - altura_texto - 10

        # Se o texto ultrapassar o topo da imagem, desenha abaixo da caixa
        if y_fundo < 0:
            y_texto = pos["y1"] + altura_texto + 5
            y_fundo = pos["y1"] + 5

        # Desenhar retângulo de fundo do texto
        cv2.rectangle(frame, 
                      (pos["x1"], y_fundo), 
                      (pos["x1"] + largura_texto, y_fundo + altura_texto + 5), 
                      cor, 
                      -1)

        # Escrever texto sobre o fundo
        cv2.putText(frame, texto, (pos["x1"], y_texto), fonte, escala_fonte, (255, 255, 255), espessura)

def detecta_faixa(frame_do_video, faixa_menor):
    # Realizar a detecção no frame
    results = model(frame_do_video)

    # 'results' é uma lista de objetos de detecção, precisamos acessar o primeiro item (que é o resultado da detecção no frame)
    result = results[0]

    frame_com_deteccoes = result.plot()  # Gera o frame com as caixas delimitadoras

    # Obter as coordenadas da caixa delimitadora dos objetos detectados
    boxes = result.boxes  # Contém as coordenadas das caixas delimitadoras

    class_names = result.names  # Contém os nomes das classes

    class_id = next((chave for chave, valor in class_names.items() if valor == "Faixa maior"), None)
    for i, box in enumerate(boxes):
        if box.cls != class_id: continue
        # Pegue a quinta caixa delimitadora (objeto 5, índice 4)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Mover para a CPU e converter para NumPy

        # Realizar o crop no objeto 5
        cropped_object = frame_do_video[int(y1):int(y2), int(x1):int(x2)]

        # Converter para HSV para segmentação de cor
        img_hsv = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2HSV)

        # Definir intervalo de cor para segmentar a faixa (verde/amarelo)
        lower_green = np.array([30, 40, 40])   # Limite inferior do verde
        upper_green = np.array([90, 255, 255]) # Limite superior do verde

        if (faixa_menor): 
            segmented = cv2.bitwise_and(cropped_object, cropped_object)
        else: 
            mask = cv2.inRange(img_hsv, lower_green, upper_green)
            segmented = cv2.bitwise_and(cropped_object, cropped_object, mask=mask)

        # Converter para escala de cinza e aplicar Canny
        gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Aplicar Transformada de Hough para detectar linhas
        linhas = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80, minLineLength=100, maxLineGap=20)

        # Criar cópia da imagem original para desenhar as linhas detectadass
        img_plot = frame_com_deteccoes.copy()

        linhas_coordenadas = []
        if linhas is not None:
            for linha in linhas:
                x1, y1, x2, y2 = linha[0]
            
                # Calcular ângulo da linha
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                # Filtrar apenas linhas com ângulo próximo ao da faixa (por exemplo, entre 70° e 110°)
                # Ajustar coordenadas da linha com base na posição do crop
                # cv2.line(img_plot, 
                #         (int(x1 + box.xyxy[0][0]), int(y1 + box.xyxy[0][1])), 
                #         (int(x2 + box.xyxy[0][0]), int(y2 + box.xyxy[0][1])), 
                #         (0, 255, 0), 2)
                linhas_coordenadas.append((int(x1 + box.xyxy[0][0]), int(y1 + box.xyxy[0][1]), int(x2 + box.xyxy[0][0]), int(y2 + box.xyxy[0][1])))

    return linhas_coordenadas

def detecta_pessoas(frame_do_video):
    # Carregar o modelo YOLO pré-treinado
    model = YOLO("../yolo11s.pt")

    results = model(frame_do_video)[0]

    pessoas_detectadas = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = results.names[class_id]

        if class_name == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Coordenadas da caixa
            pessoas_detectadas.append((x1, y1, x2, y2))
    return pessoas_detectadas

def grava_video():
    # Abrir o vídeo
    video_path = "0_te21.mp4"
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi carregado corretamente
    if not cap.isOpened():
        print("Erro ao carregar o vídeo")

    # Pegar largura, altura e fps do vídeo original
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Criar o writer para salvar o vídeo com as anotações
    output_path = "video_processado.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (largura, altura))

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Fim do vídeo")
            break

        coordenadas_objetos = detecta_objetos(frame)
        desenha_retangulos_objetos(coordenadas_objetos, frame)

        coordenadas_linhas = detecta_faixa(frame, faixa_menor=False)
        for (x1, y1, x2, y2) in coordenadas_linhas:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        coordenadas_pessoas = detecta_pessoas(frame)
        for (x1, y1, x2, y2) in coordenadas_pessoas:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Pessoa", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)

        # Mostrar na tela
        # cv2.imshow("Deteccao", frame)

        # Salvar frame no novo vídeo
        video_writer.write(frame)

        # Sair com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def testa_antes_de_gravar():
    # Abrir o vídeo
    video_path = "0_te21.mp4"
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi carregado corretamente
    if not cap.isOpened():
        print("Erro ao carregar o vídeo")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Fim do vídeo")
            break

        coordenadas_objetos = detecta_objetos(frame)
        desenha_retangulos_objetos(coordenadas_objetos, frame)

        coordenadas_linhas = detecta_faixa(frame, faixa_menor=False)
        for (x1, y1, x2, y2) in coordenadas_linhas:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        coordenadas_pessoas = detecta_pessoas(frame)
        for (x1, y1, x2, y2) in coordenadas_pessoas:
            # Desenhar o retângulo da pessoa
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Escrever o texto "Pessoa" acima da caixa
            cv2.putText(frame, "Pessoa", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)
            
        cv2.imshow("Deteccao", frame)

        # Sair com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# grava_video()
testa_antes_de_gravar()