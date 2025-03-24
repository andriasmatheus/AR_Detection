import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = "runs/detect/train9/weights/best.pt"
CLASSE_DESEJADA = 2  # Altere conforme necessário
VIDEO_PATH = "dataset/raw/train/1_tr1.mp4"
frame_interval = 5  # Processa um quadro a cada 5 quadros


def region_growing(frame):
    """Aplica a segmentação Region Growing no frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray, dtype=np.uint8)
    
    seed_point = (gray.shape[0] // 2, gray.shape[1] // 2)
    x, y = seed_point
    mask[x, y] = 255
    points = [seed_point]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    region_value = gray[x, y]

    while points:
        cx, cy = points.pop()
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < gray.shape[0] and 0 <= ny < gray.shape[1]:
                if abs(int(gray[nx, ny]) - int(region_value)) <= 10 and mask[nx, ny] == 0:
                    mask[nx, ny] = 255
                    points.append((nx, ny))

    return mask


def process_video():
    """Captura o vídeo, detecta o objeto desejado e aplica segmentação Region Growing."""
    print("Teste")
    model = YOLO(MODEL_PATH).to("cuda")
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Sai do loop quando o vídeo termina

        frame_resized = cv2.resize(frame, (640, 480))
        results = model(frame_resized)  # Faz a detecção no frame redimensionado
        cropped = None

        for box in results[0].boxes:
            class_id = int(box.cls.cpu().numpy())
            if class_id == CLASSE_DESEJADA:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                h, w, _ = frame.shape
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                cropped = frame[y1:y2, x1:x2]
                break

        if frame_count % frame_interval == 0 and cropped is not None:
            segmented = region_growing(cropped)
            combined = np.hstack((cropped, cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)))
            cv2.imshow("Detecção e Segmentação", combined)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


# Executa o processamento do vídeo
process_video()
