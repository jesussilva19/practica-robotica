import cv2
from ultralytics import YOLO

# Elige uno:
MODEL_PATH = "yolo11n-pose.pt"   # opción nueva (YOLO11)
# MODEL_PATH = "yolov8n-pose.pt" # opción estable (YOLOv8)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)  # 0 = webcam
assert cap.isOpened(), "No se pudo abrir la cámara"

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # inferencia pose (conf por defecto; puedes subir/bajar)
    results = model.predict(frame, verbose=False)
    annotated = results[0].plot()  # dibuja skeleton/keypoints

    cv2.imshow("YOLO Pose - Live", annotated)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
