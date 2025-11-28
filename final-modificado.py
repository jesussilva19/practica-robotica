"""
PRACTICA 3 - ROBOTICA

Integrantes:
    - Miguel López López - miguel.lopezl@udc.es
    - Jesús Silva Vicente- jesus.silva@udc.es
"""


from robobopy_videostream.RoboboVideo import RoboboVideo
from robobopy.Robobo import Robobo
from ultralytics import YOLO
from stable_baselines3 import PPO

import cv2
import sys
import time
import numpy as np
import os

# ================== CONFIGURACIÓN ==================
IP = "172.20.10.3"        # IP del móvil con la app del Robobo

POSE_MODEL_PATH = "yolo11n-pose.pt"   # modelo de pose 
DET_MODEL_PATH  = "yolo11n.pt"        # modelo detección COCO 

IMG_SIZE    = 160        # tamaño de imagen para YOLO 
POSE_CONF   = 0.5
PHONE_CONF  = 0.5        # umbral para activar PPO

# PPO (usa el modelo de la práctica 1)
BASE_DIR = os.path.dirname(__file__)
PPO_MODEL_PATH = os.path.join(BASE_DIR, "practica3", "best_model.zip")
MAX_PPO_STEPS = 150

SPEED_FWD       = 5
TURN_SPEED      = 5
CMD_TIME_SHORT  = 2.0
CMD_TIME_LONG   = 4.0
CMD_TIME_TURN180 = 3.0

WRIST_MARGIN   = 20
TORSO_X_OFFSET = 60

# Índices COCO para pose
L_SHOULDER = 5
R_SHOULDER = 6
L_WRIST = 9
R_WRIST = 10

DEVICE = 'cpu'      # 'cpu' o 'cuda' para GPU NVIDIA


def to_numpy(tensor_or_array):
    """Convierte input de Torch o Listas a Numpy array."""
    if hasattr(tensor_or_array, 'cpu'):
        return tensor_or_array.cpu().numpy()
    return np.asarray(tensor_or_array)


def get_main_person_keypoints(result):
    """
    Extrae los keypoints de la persona con mayor confianza.
    Retorna: np.array (17, 2) o None
    """
    if result is None or not hasattr(result, "keypoints"):
        return None

    # Obtenemos datos y convertimos a numpy
    kps_obj = result.keypoints
    xy = to_numpy(kps_obj.xy)    # Shape esperada: (N, 17, 2)
    conf = to_numpy(kps_obj.conf) # Shape esperada: (N, 17)

    # Validaciones de estructura
    if xy.ndim != 3 or xy.shape[0] == 0:
        return None

    # Lógica de selección de persona. Si no hay confianza, asumimos la primera persona (index 0)
    best_idx = 0
    if conf is not None and conf.ndim == 2:
        # Promedio de confianza por persona ignorando NaNs
        mean_conf = np.nanmean(conf, axis=1)
        if not np.all(np.isnan(mean_conf)):
            best_idx = int(np.nanargmax(mean_conf))

    person_kps = xy[best_idx]
    
    # Validación de keypoints mínimos
    if person_kps.shape[0] < 11: 
        return None
        
    return person_kps


def gesture_from_keypoints(kp):
    """Determina el gesto basado en la posición relativa de muñecas y hombros."""
    if kp is None or kp.shape[0] <= R_WRIST: 
        return "STOP"

    # Extracción de coordenadas
    try:
        lx, ly = kp[L_WRIST]
        rx, ry = kp[R_WRIST]
        lsh_x, lsh_y = kp[L_SHOULDER]
        rsh_x, rsh_y = kp[R_SHOULDER]
    except IndexError:
        return "STOP"

    # Validación de datos
    if any(c == 0 for c in [lx, ly, rx, ry, lsh_x, lsh_y, rsh_x, rsh_y]):
        return "STOP"

    # Cálculo de estados 
    left_up  = ly < (lsh_y - WRIST_MARGIN)
    right_up = ry < (rsh_y - WRIST_MARGIN)
    
    # Diferencia del torso 
    torso_diff = rsh_x - lsh_x # >0 hombro derecho más a la derecha

    # Árbol de decisión
    if left_up and right_up:
        return "FORWARD"
    if right_up:
        return "TURN_RIGHT"
    if left_up:
        return "TURN_LEFT"
    if torso_diff < -TORSO_X_OFFSET:
        return "TURN_LEFT"
    if torso_diff >  TORSO_X_OFFSET:
        return "TURN_RIGHT"
    
    return "STOP"
   




def compute_phone_state(det_results, frame_width):
    """
    A partir de las detecciones de YOLO en la cámara del Robobo,
    devolvemos:
      - state: entero en [0,13] (como si fuera el estado del blob).
      - seen: True si ha visto un 'bottle'.
    Usamos la posición X del centro del bounding box.
    
    """
    time.sleep(2)  # evitar problemas de latencia 
    best_conf = 0.0
    best_xcenter = None

    for box in det_results[0].boxes:
        cls_id = int(box.cls)
        cls_name = det_results[0].names[cls_id]
        conf = float(box.conf)
        if cls_name == "bottle" and conf > 0.5 and conf > best_conf:
            best_conf = conf
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            best_xcenter = (x1 + x2) / 2.0

    if best_xcenter is None:
        # No hay móvil visible -> estado "no visible"
        return 13, False

    # Normalizar X a [0,1]
    norm_x = best_xcenter / float(frame_width)
    
    # Mapear a 14 estados discretos [0..13]
    state = int(norm_x * 14)
    if state < 0:
        state = 0
    if state > 13:
        state = 13
    return state, True


def apply_ppo_action(rob, action):
    """
    Aplica una acción {0..5} como en el robot real.
    """
    a = int(action)
   
    if a == 0:  # Avanzar recto
        rob.moveWheelsByTime(SPEED_FWD, SPEED_FWD, CMD_TIME_SHORT)
    elif a == 1:  # Girar izquierda leve
        rob.moveWheelsByTime(0, 2, CMD_TIME_SHORT)
    elif a == 2:  # Girar derecha leve
        rob.moveWheelsByTime(2, 0, CMD_TIME_SHORT)
    elif a == 3:  # Girar izquierda fuerte
        rob.moveWheelsByTime(0, 3, CMD_TIME_LONG)
    elif a == 4:  # Girar derecha fuerte
        rob.moveWheelsByTime(3, 0, CMD_TIME_LONG)
    elif a == 5:  # Giro 180°
        rob.moveWheelsByTime(TURN_SPEED, -TURN_SPEED, CMD_TIME_TURN180)
    else:
        # Por si el modelo da algo raro
        rob.stopMotors()


# ================== LIMPIEZA ==================
def cleanup(rob, video, cap, exit_code=0):
    try:
        if rob is not None:
            rob.stopMotors()
    except:
        pass
    try:
        if video is not None:
            video.disconnect()
    except:
        pass
    try:
        if rob is not None:
            rob.disconnect()
    except:
        pass
    try:
        if cap is not None and cap.isOpened():
            cap.release()
    except:
        pass
    cv2.destroyAllWindows()
    print("Exit cleanly.")
    sys.exit(exit_code)


def get_frames(video_obj, cap_obj):
    """Lee ambas cámaras. Devuelve None si fallan."""
    # Cámara Móvil (Robobo)
    frame_phone = None
    try:
        frame_phone, _, _, _ = video_obj.getImageWithMetadata()
    except Exception:
        pass # frame_phone se queda en None

    # Cámara PC (Webcam)
    ok, frame_pc = cap_obj.read()
    if ok:
        cv2.flip(frame_pc, 1, frame_pc) # Espejo
    else:
        frame_pc = None

    return frame_phone, frame_pc


def check_switch_to_ppo(det_model, frame, conf_threshold):
    """Verifica si aparece una botella con alta confianza."""
    if frame is None: return False
    
    # Inferencia ligera
    results = det_model.predict(frame, imgsz=IMG_SIZE, conf=conf_threshold, verbose=False, device=DEVICE)
    
    # Buscamos 'bottle' en los resultados
    for box in results[0].boxes:
        if results[0].names[int(box.cls)] == "bottle" and float(box.conf) > conf_threshold:
            return True
    return False


# ================== MAIN ==================
def main():
    # Conexión al ROBOT REAL
    rob = Robobo(IP)
    rob.connect()
    print(f"Conectado a Robobo en {IP}")

    # Stream de la cámara del móvil
    video = RoboboVideo(IP)
    video.connect()
    rob.startStream()
    print("Streaming de la cámara del móvil iniciado.")

    # Webcam del PC para los gestos
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Resolución para máxima fluidez
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Resolución para máxima fluidez
    cap.set(cv2.CAP_PROP_FPS, 15)            # Limitar FPS
    
    assert cap.isOpened(), "No se pudo abrir la webcam del PC"

    # Modelos YOLO
    pose_model = YOLO(POSE_MODEL_PATH)
    det_model  = YOLO(DET_MODEL_PATH)

    # Cargar PPO
    try:
        ppo_model = PPO.load(PPO_MODEL_PATH)
        print(f"Modelo PPO cargado desde {PPO_MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo PPO: {e}")
        cleanup(rob, video, cap, exit_code=1)

    mode = "TELEOP"
    ppo_steps = 0

    print("Teleoperación lista.")
    print("Gestos: " \
    "\n Ambos brazos arriba = adelante; " \
    "\n Brazo izquierdo abajo, brazo derecho arriba = girar izquierda;" \
    "\n Brazo izquierdo arriba, brazo derecho abajo = girar derecha;"
    "\n Sin gesto = stop.")
    print("Al detectar 'bottle' en la cámara del Robobo con conf>0.8, pasará a MODO PPO.\n")

    try:
        while True:
            # ---------- 1) LEER CÁMARAS ----------
            frame_phone, frame_pc = get_frames(video, cap)
            

            # ---------- 2) MODO TELEOP ----------
            if mode == "TELEOP":
                if frame_pc is None: continue

                pose_results = pose_model.predict(frame_pc, imgsz=IMG_SIZE, conf=POSE_CONF, verbose=False, device=DEVICE)
                r_pose = pose_results[0]
                kp = get_main_person_keypoints(r_pose)
                cmd = gesture_from_keypoints(kp)

                
                # Si se ha visto el móvil con suficiente confianza -> cambio a PPO
                if check_switch_to_ppo(det_model, frame_phone, PHONE_CONF):
                    print("\n>> bottle DETECTADO con suficiente confianza.")
                    print("CAMBIO DE MODO: TELEOP ➜ PPO")
                    mode = "PPO"
                    ppo_steps = 0
                    continue  # siguiente iteración ya en modo PPO

                
               # Control continuo del robobo
                if cmd == "FORWARD":
                    rob.moveWheels(SPEED_FWD, SPEED_FWD)
                elif cmd == "TURN_RIGHT":
                    rob.moveWheels(0, SPEED_FWD)
                elif cmd == "TURN_LEFT":
                    rob.moveWheels(SPEED_FWD, 0)
                else:  # STOP o desconocido
                    rob.stopMotors()

                # Mostrar webcam con pose
                annotated_pc = r_pose.plot()
                cv2.putText(
                    annotated_pc,
                    f"MODE: {mode} CMD: {cmd}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Teleop Robobo REAL (YOLO-Pose)", annotated_pc)

            # ---------- 3) MODO PPO ----------
            else:  # mode == "PPO"
                rob.moveTiltTo(90, 50)
                if frame_phone is None: 
                    rob.stopMotors()
                    continue

                # Inferencia Estado
                det_results = det_model.predict(frame_phone, imgsz=IMG_SIZE, conf=0.5, verbose=False, device=DEVICE)
                phone_state, _ = compute_phone_state(det_results, frame_phone.shape[1])

                #Inferencia Acción
                action, _ = ppo_model.predict(np.array([phone_state]), deterministic=True)   # Estado ya calculado antes: phone_state (0..13)
                apply_ppo_action(rob, action)
                ppo_steps += 1

                # Límite de pasos PPO
                if ppo_steps >= MAX_PPO_STEPS:
                    print("MAX_PPO_STEPS alcanzado. Terminando.")
                    break

                # Mostrar frame con info PPO
                cv2.putText(
                    frame_phone,
                    f"MODE: {mode} state:{phone_state} action:{int(action)}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Camara Robobo + YOLO", frame_phone)


            # ---------- 4) TECLA ESC PARA SALIR ----------
            if cv2.waitKey(1) & 0xFF == 27: break

    except KeyboardInterrupt:
        print("\nDetenido por usuario.")
    finally:
        cleanup(rob, video, cap, exit_code=0)


if __name__ == "__main__":
    main()
