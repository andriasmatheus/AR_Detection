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

    if faixa_menor == True:
        classe_name = "Faixa menor"
    else: 
        classe_name = "Faixa maior"
    class_id = next((chave for chave, valor in class_names.items() if valor == classe_name), None)
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
                # if faixa_menor and angle < 0 or not faixa_menor and 60 <= angle <= 80:
                if faixa_menor and -80 <= angle < 0 or not faixa_menor and 60 <= angle <= 80:
                    #Ajustar coordenadas da linha com base na posição do crop
                    linhas_coordenadas.append((int(x1 + box.xyxy[0][0]), int(y1 + box.xyxy[0][1]), int(x2 + box.xyxy[0][0]), int(y2 + box.xyxy[0][1])))

    return linhas_coordenadas

def detecta_pessoas(frame_do_video):
    # Carregar o modelo YOLO pré-treinado
    model = YOLO("yolo11s.pt")

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

        coordenadas_linhas_menor = detecta_faixa(frame, faixa_menor=True)
        for (x1, y1, x2, y2) in coordenadas_linhas_menor:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        coordenadas_linhas_maior = detecta_faixa(frame, faixa_menor=False)
        for (x1, y1, x2, y2) in coordenadas_linhas_maior:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        coordenadas_objetos = detecta_objetos(frame)
        desenha_retangulos_objetos(coordenadas_objetos, frame)

        coordenadas_pessoas = detecta_pessoas(frame)
        for (x1_pessoa, y1_pessoa, x2_pessoa, y2_pessoa) in coordenadas_pessoas:
            # Verificar se a pessoa está entre as faixas (fora da faixa de segurança)
            pessoa_fora_faixa = False

            for (x1_faixa_menor, y1_faixa_menor, x2_faixa_menor, y2_faixa_menor) in coordenadas_linhas_menor:
                for (x1_faixa_maior, y1_faixa_maior, x2_faixa_maior, y2_faixa_maior) in coordenadas_linhas_maior:
                    # Verificar se a pessoa está à direita da faixa menor e à esquerda da faixa maior
                    if x1_pessoa > x2_faixa_menor and x2_pessoa < x1_faixa_maior:
                        cv2.putText(frame, "Pessoa fora da faixa de segurança", (x1_pessoa, y1_pessoa - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        pessoa_fora_faixa = True

            # Desenhar o retângulo da pessoa (em vermelho se fora da faixa de segurança)
            cor = (0, 255, 0)  # Cor padrão (verde)
            if pessoa_fora_faixa:
                cor = (0, 0, 255)  # Cor vermelha se fora da faixa de segurança

            cv2.rectangle(frame, (x1_pessoa, y1_pessoa), (x2_pessoa, y2_pessoa), cor, 2)

            # Escrever o texto "Pessoa" acima da caixa
            if pessoa_fora_faixa != True:
                cv2.putText(frame, "Pessoa em posição segura", (x1_pessoa, y1_pessoa - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, cor, 2)

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

        coordenadas_linhas_menor = detecta_faixa(frame, faixa_menor=True)
        for (x1, y1, x2, y2) in coordenadas_linhas_menor:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        coordenadas_linhas_maior = detecta_faixa(frame, faixa_menor=False)
        for (x1, y1, x2, y2) in coordenadas_linhas_maior:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        coordenadas_objetos = detecta_objetos(frame)
        desenha_retangulos_objetos(coordenadas_objetos, frame)

        coordenadas_pessoas = detecta_pessoas(frame)
        for (x1_pessoa, y1_pessoa, x2_pessoa, y2_pessoa) in coordenadas_pessoas:
            # Verificar se a pessoa está entre as faixas (fora da faixa de segurança)
            pessoa_fora_faixa = False

            for (x1_faixa_menor, y1_faixa_menor, x2_faixa_menor, y2_faixa_menor) in coordenadas_linhas_menor:
                for (x1_faixa_maior, y1_faixa_maior, x2_faixa_maior, y2_faixa_maior) in coordenadas_linhas_maior:
                    # Verificar se a pessoa está à direita da faixa menor e à esquerda da faixa maior
                    if x1_pessoa > x2_faixa_menor and x2_pessoa < x1_faixa_maior:
                        cv2.putText(frame, "Pessoa fora da faixa de segurança", (x1_pessoa, y1_pessoa - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        pessoa_fora_faixa = True

            # Desenhar o retângulo da pessoa (em vermelho se fora da faixa de segurança)
            cor = (0, 255, 0)  # Cor padrão (verde)
            if pessoa_fora_faixa:
                cor = (0, 0, 255)  # Cor vermelha se fora da faixa de segurança

            cv2.rectangle(frame, (x1_pessoa, y1_pessoa), (x2_pessoa, y2_pessoa), cor, 2)

            # Escrever o texto "Pessoa" acima da caixa
            if pessoa_fora_faixa != True:
                cv2.putText(frame, "Pessoa em posição segura", (x1_pessoa, y1_pessoa - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, cor, 2)

        # Mostrar o vídeo com as anotações
        cv2.imshow("Deteccao", frame)

        # Sair com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()



## Gera e grava o vídeo dos histogramas
def gera_histogramas():
    # Carregar o modelo pré-treinado
    model_path = "runs/detect/train10/weights/best.pt"
    model = YOLO(model_path)

    # Abrir o vídeo
    video_path = "5_te12.mp4"
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi carregado corretamente
    if not cap.isOpened():
        print("Erro ao carregar o vídeo")
        return

    # Configuração do VideoWriter para gravar o vídeo dos histogramas
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para gravar o vídeo
    out = cv2.VideoWriter('video_dos_histogramas.mp4', fourcc, 20.0, (640, 480))  # Ajuste o tamanho conforme necessário

    # Contador de frames
    frame_counter = 0
    hist_image_resized = None  # Variável para armazenar a última imagem de histograma

    while True:
        # Captura frame por frame
        ret, frame = cap.read()

        if not ret:
            print("Fim do vídeo")
            break

        frame_counter += 1  # Incrementar contador de frames

        # Realizar a detecção no frame
        results = model(frame)
        
        # 'results' é uma lista de objetos de detecção, precisamos acessar o primeiro item (que é o resultado da detecção no frame)
        result = results[0]

        # Obter as coordenadas da caixa delimitadora do segundo objeto detectado (objeto 5)
        boxes = result.boxes  # Contém as coordenadas das caixas delimitadoras

        if len(boxes) > 4:  # Verificar se pelo menos 5 objetos foram detectados
            # Pegue a quinta caixa delimitadora (objeto 5, índice 4)
            box = boxes[4]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Mover para a CPU e converter para NumPy

            # Realizar o crop no objeto 5
            cropped_object = frame[int(y1):int(y2), int(x1):int(x2)]

            # Converter o crop para escala de cinza
            gray_crop = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2GRAY)

            # Calcular gradientes usando Sobel
            grad_x = cv2.Sobel(gray_crop, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_crop, cv2.CV_64F, 0, 1, ksize=3)

            # Calcular magnitude e direção do gradiente
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direcao = np.arctan2(grad_y, grad_x) * (180 / np.pi)  # Converter para graus

            # Selecionar pixels que estão na borda (tem pelo menos um vizinho diferente)
            bordas = (gray_crop > 0) & (cv2.morphologyEx(gray_crop, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8)) > 0)

            # Filtrar direções apenas nos pixels da borda
            direcoes_borda = direcao[bordas]

            # Calcular o histograma de tons de cinza
            gray_hist = cv2.calcHist([gray_crop], [0], None, [256], [0, 256])
            gray_hist = cv2.normalize(gray_hist, gray_hist).flatten()  # Normalizar e achatar para exibição

            # Criar figura com subplots para exibir os resultados
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # Exibir imagem original (crop) em escala de cinza
            axs[0].imshow(gray_crop, cmap='gray')
            axs[0].set_title("Imagem em Escala de Cinza do Crop")
            axs[0].axis("off")

            if frame_counter == 1:
                direcoes_borda_anterior = direcoes_borda
                gray_hist_anterior = gray_hist

            # Atualizar histogramas apenas a cada 5 frames
            if frame_counter % 5 == 0:
                # Exibir histograma das direções dos gradientes
                axs[1].hist(direcoes_borda, bins=30, range=(0, 180), edgecolor='black')
                axs[1].set_xlabel("Direção do Gradiente (graus)")
                axs[1].set_ylabel("Frequência")
                axs[1].set_title("Histograma das Direções do Gradiente na Borda")

                # Exibir histograma de tons de cinza
                axs[2].plot(range(256), gray_hist, color='gray')
                axs[2].set_xlabel("Intensidade do Tom de Cinza")
                axs[2].set_ylabel("Frequência")
                axs[2].set_title("Histograma de Tons de Cinza")
                direcoes_borda_anterior = direcoes_borda
                gray_hist_anterior = gray_hist
            else:
                # Exibir histograma das direções dos gradientes
                axs[1].hist(direcoes_borda_anterior, bins=30, range=(0, 180), edgecolor='black')
                axs[1].set_xlabel("Direção do Gradiente (graus)")
                axs[1].set_ylabel("Frequência")
                axs[1].set_title("Histograma das Direções do Gradiente na Borda")

                # Exibir histograma de tons de cinza
                axs[2].plot(range(256), gray_hist_anterior, color='gray')
                axs[2].set_xlabel("Intensidade do Tom de Cinza")
                axs[2].set_ylabel("Frequência")
                axs[2].set_title("Histograma de Tons de Cinza")                


            plt.tight_layout()

            # Salvar a figura como uma imagem temporária
            plt.savefig("temp_histograma.png", bbox_inches='tight')
            plt.close(fig)

            # Ler a imagem salva e redimensionar para o tamanho do vídeo
            hist_image_resized = cv2.imread("temp_histograma.png")
            hist_image_resized = cv2.resize(hist_image_resized, (640, 480))

        # Exibir o vídeo em tons de cinza em tempo real
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_resized = cv2.resize(gray_frame, (640, 480))  # Ajuste para o tamanho do vídeo
        cv2.imshow("Vídeo em Tons de Cinza", gray_frame_resized)

        # Escrever o quadro de histogramas no vídeo
        if hist_image_resized is not None:
            out.write(hist_image_resized)

        # Quebra o loop se pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar o vídeo e fechar as janelas
    cap.release()
    out.release()  # Fechar o arquivo de vídeo
    cv2.destroyAllWindows()

# gera_histogramas()
grava_video()
# testa_antes_de_gravar()