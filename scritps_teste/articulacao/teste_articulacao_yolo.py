from ultralytics import YOLO
import cv2
import time

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model

# Predict with the model
# results = model("dataset/raw/capture.png")  # predict on an image
results = model("dataset/raw/train/1_tr69.mp4")  # predict on a video
test = True

# Definir a largura e altura da janela de exibição
window_width, window_height = 1280, 720
cv2.namedWindow('YOLO11 Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO11 Detection', window_width, window_height)

for result in results:
    result_img = result.plot()


    # Exibir o frame com as detecções e poses
    cv2.imshow('YOLO11 Detection', result_img)

    # Pressione 'q' para sair do loop de exibição
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop = True
        break
    time.sleep(0.1)

cv2.destroyAllWindows()

