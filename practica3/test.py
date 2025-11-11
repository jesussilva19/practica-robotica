from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("yolo11s-pose.pt")  # load an official model

# Nombres de los keypoints según el formato COCO
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

# Índices de los keypoints de los brazos
ARM_KEYPOINTS = {
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10
}

def draw_arms(frame, keypoints, confidence_threshold=0.5):
    """
    Dibuja los brazos y muestra información de posición
    """
    if keypoints is None or len(keypoints) == 0:
        return frame
    
    # keypoints tiene forma (num_keypoints, 3) donde 3 = (x, y, confidence)
    kpts = keypoints[0]  # Tomar la primera persona detectada
    
    # Definir colores
    color_left = (0, 255, 0)   # Verde para brazo izquierdo
    color_right = (255, 0, 0)  # Azul para brazo derecho
    
    # Dibujar brazo izquierdo
    left_points = []
    for name in ['left_shoulder', 'left_elbow', 'left_wrist']:
        idx = ARM_KEYPOINTS[name]
        if idx < len(kpts):
            x, y, conf = kpts[idx]
            if conf > confidence_threshold:
                left_points.append((int(x), int(y)))
                cv2.circle(frame, (int(x), int(y)), 5, color_left, -1)
                cv2.putText(frame, f"{name.split('_')[1]}", 
                           (int(x) + 10, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_left, 2)
    
    # Conectar puntos del brazo izquierdo
    for i in range(len(left_points) - 1):
        cv2.line(frame, left_points[i], left_points[i+1], color_left, 2)
    
    # Dibujar brazo derecho
    right_points = []
    for name in ['right_shoulder', 'right_elbow', 'right_wrist']:
        idx = ARM_KEYPOINTS[name]
        if idx < len(kpts):
            x, y, conf = kpts[idx]
            if conf > confidence_threshold:
                right_points.append((int(x), int(y)))
                cv2.circle(frame, (int(x), int(y)), 5, color_right, -1)
                cv2.putText(frame, f"{name.split('_')[1]}", 
                           (int(x) + 10, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_right, 2)
    
    # Conectar puntos del brazo derecho
    for i in range(len(right_points) - 1):
        cv2.line(frame, right_points[i], right_points[i+1], color_right, 2)
    
    return frame

def get_arm_positions(keypoints, confidence_threshold=0.5):
    """
    Extrae las posiciones de los brazos
    """
    if keypoints is None or len(keypoints) == 0:
        return None
    
    kpts = keypoints[0]
    positions = {}
    
    for name, idx in ARM_KEYPOINTS.items():
        if idx < len(kpts):
            x, y, conf = kpts[idx]
            if conf > confidence_threshold:
                positions[name] = {
                    'x': float(x),
                    'y': float(y),
                    'confidence': float(conf)
                }
    
    return positions

def main():
    # Abrir la cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
    
    print("Iniciando detección de pose en tiempo real...")
    print("Presiona 'q' para salir")
    print("Presiona 'p' para imprimir las posiciones actuales")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: No se pudo leer el frame")
            break
        
        # Realizar la predicción
        results = model(frame, verbose=False)
        
        # Obtener keypoints
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
            
            # Dibujar los brazos
            frame = draw_arms(frame, keypoints)
            
            # Obtener posiciones (opcional, para debugging)
            positions = get_arm_positions(keypoints)
            
            # Mostrar información en pantalla
            if positions:
                y_offset = 30
                cv2.putText(frame, "Brazos detectados:", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                y_offset += 30
                for joint_name, pos in positions.items():
                    text = f"{joint_name}: ({pos['x']:.0f}, {pos['y']:.0f})"
                    cv2.putText(frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 25
        
        # Mostrar el frame
        cv2.imshow('Detección de Brazos - YOLO Pose', frame)
        
        # Controles de teclado
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            # Imprimir posiciones en consola
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints = results[0].keypoints.data.cpu().numpy()
                positions = get_arm_positions(keypoints)
                if positions:
                    print("\n=== Posiciones de los brazos ===")
                    for joint_name, pos in positions.items():
                        print(f"{joint_name}: x={pos['x']:.2f}, y={pos['y']:.2f}, conf={pos['confidence']:.2f}")
                    print("================================\n")
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
