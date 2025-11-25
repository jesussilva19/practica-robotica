from robobopy_videostream.RoboboVideo import RoboboVideo
from robobopy.Robobo import Robobo
from ultralytics import YOLO
from stable_baselines3 import PPO

import cv2
import sys
import time
import numpy as np
import os

# ================== CONFIG ==================
IP = "172.20.10.2"          # IP del móvil con la app del Robobo

# Modelos YOLO
POSE_MODEL_PATH = "yolo11n-pose.pt"   # modelo de pose
DET_MODEL_PATH  = "yolo11n.pt"        # modelo detección COCO

IMG_SIZE    = 640
POSE_CONF   = 0.5
PHONE_CONF  = 0.5        # umbral para activar PPO

# PPO (usa el modelo de la práctica 1)
BASE_DIR = os.path.dirname(__file__)
PPO_MODEL_PATH = os.path.join(BASE_DIR, "practica3", "best_model.zip")
MAX_PPO_STEPS = 150

# Movimiento robot (coherente con RoboboEnv)
SPEED_FWD       = 5
TURN_SPEED      = 5
CMD_TIME_SHORT  = 2.0
CMD_TIME_LONG   = 4.0
CMD_TIME_TURN180 = 3.0

WRIST_MARGIN   = 20
TORSO_X_OFFSET = 60

# Índices COCO para pose
NOSE = 0
L_SHOULDER = 5
R_SHOULDER = 6
L_WRIST = 9
R_WRIST = 10


# ================== HELPERS GESTOS ==================
def get_main_person_keypoints(result):
    kps = getattr(result, "keypoints", None)
    if kps is None:
        return None
    xy = kps.xy
    try:
        xy = xy.cpu().numpy()
    except Exception:
        xy = np.asarray(xy)

    if xy.ndim != 3 or xy.shape[-1] != 2 or xy.shape[1] < 11 or xy.shape[0] == 0:
        return None

    # Elegir persona con mayor confianza media
    idx = 0
    conf = getattr(kps, "conf", None)
    if conf is not None:
        c = conf
        try:
            c = c.cpu().numpy()
        except Exception:
            c = np.asarray(c)
        if c.ndim == 2 and c.shape[0] == xy.shape[0]:
            m = np.nanmean(c, axis=1)
            if not np.all(np.isnan(m)):
                idx = int(np.nanargmax(m))

    kp = xy[idx]
    if kp.shape[0] >= 11 and np.isfinite(kp[:11]).all():
        return kp[:17] if kp.shape[0] >= 17 else kp
    return None


def gesture_from_keypoints(kp):
    try:
        if kp is None or kp.shape[0] <= R_WRIST:
            return "STOP"

        lx, ly = kp[L_WRIST]
        rx, ry = kp[R_WRIST]
        lsh_x, lsh_y = kp[L_SHOULDER]
        rsh_x, rsh_y = kp[R_SHOULDER]

        vals = np.array([lx, ly, rx, ry, lsh_x, lsh_y, rsh_x, rsh_y], dtype=float)
        if not np.isfinite(vals).all():
            return "STOP"

        arm_right_up = ry < (rsh_y - WRIST_MARGIN)
        arm_left_up  = ly < (lsh_y - WRIST_MARGIN)
        torso_delta  = rsh_x - lsh_x  # >0 hombro derecho más a la derecha

        if arm_left_up and arm_right_up:
            return "FORWARD"
        if arm_right_up and not arm_left_up:
            return "TURN_RIGHT"
        if arm_left_up and not arm_right_up:
            return "TURN_LEFT"
        if torso_delta < -TORSO_X_OFFSET:
            return "TURN_LEFT"
        if torso_delta >  TORSO_X_OFFSET:
            return "TURN_RIGHT"
        return "STOP"
    except Exception:
        return "STOP"


# ================== HELPERS PPO ==================
def compute_phone_state(det_results, frame_width):
    """
    A partir de las detecciones de YOLO en la cámara del Robobo,
    devolvemos:
      - state: entero en [0,13] (como si fuera el estado del blob).
      - seen: True si ha visto un 'backpack'.
    Usamos la posición X del centro del bounding box.
    """
    best_conf = 0.0
    best_xcenter = None
    for box in det_results[0].boxes:
        cls_id = int(box.cls)
        cls_name = det_results[0].names[cls_id]
        conf = float(box.conf)
        if cls_name == "backpack" and conf > 0.5 and conf > best_conf:
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
    Aplica una acción {0..5} como en tu RoboboEnv.step, pero en el robot real.
    """
    a = int(action)
    if a == 0:  # Avanzar recto
        rob.moveWheelsByTime(SPEED_FWD, SPEED_FWD, CMD_TIME_SHORT)
    elif a == 1:  # Girar izquierda leve
        rob.moveWheelsByTime(0, SPEED_FWD, CMD_TIME_SHORT)
    elif a == 2:  # Girar derecha leve
        rob.moveWheelsByTime(SPEED_FWD, 0, CMD_TIME_SHORT)
    elif a == 3:  # Girar izquierda fuerte
        rob.moveWheelsByTime(0, SPEED_FWD, CMD_TIME_LONG)
    elif a == 4:  # Girar derecha fuerte
        rob.moveWheelsByTime(SPEED_FWD, 0, CMD_TIME_LONG)
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
    print("Gestos: ambos brazos=adelante; brazo izq=izq; brazo dcho=dcha; sin gesto=stop.")
    print("Al detectar 'backpack' en la cámara del Robobo con conf>0.8, pasará a MODO PPO.\n")

    try:
        while True:
            # ---------- 1) LEER CÁMARA DEL MÓVIL ----------
            annotated_phone = None
            frame_phone = None
            try:
                frame_phone, ts, sync_id, frame_id = video.getImageWithMetadata()
            except Exception:
                frame_phone = None

            phone_state = 13
            phone_seen_for_switch = False

            if frame_phone is not None:
                det_results = det_model.predict(frame_phone, imgsz=IMG_SIZE, conf=0.5, verbose=False)
                annotated_phone = det_results[0].plot()
                h, w, _ = frame_phone.shape

                # Estado PPO basado en posición del móvil
                phone_state, phone_seen = compute_phone_state(det_results, w)

                # Condición de cambio TELEOP -> PPO: detección de backpack con alta conf
                for box in det_results[0].boxes:
                    cls_id = int(box.cls)
                    cls_name = det_results[0].names[cls_id]
                    conf = float(box.conf)
                    if cls_name == "backpack" and conf > PHONE_CONF:
                        phone_seen_for_switch = True
                        break

            # ---------- 2) MODO TELEOP ----------
            if mode == "TELEOP":
                # Si se ha visto el móvil con suficiente confianza -> cambias a PPO
                if phone_seen_for_switch:
                    print("\nBACKPACK DETECTADO con suficiente confianza.")
                    print("CAMBIO DE MODO: TELEOP ➜ PPO")
                    mode = "PPO"
                    # Pequeña pausa para que lo veas
                    time.sleep(1.5)
                    continue  # siguiente iteración ya en modo PPO

                # Si seguimos en TELEOP, usamos gestos
                ok, frame_pc = cap.read()
                if not ok:
                    # aunque no haya frame de la webcam, seguimos refrescando ventanas
                    if annotated_phone is not None:
                        cv2.imshow("Camara Robobo + YOLO", annotated_phone)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                pose_results = pose_model.predict(frame_pc, imgsz=IMG_SIZE, conf=POSE_CONF, verbose=False)
                r_pose = pose_results[0]
                kp = get_main_person_keypoints(r_pose)
                cmd = gesture_from_keypoints(kp)

                print(f"[TELEOP] CMD: {cmd}")

                # Ejecutar comando en el ROBOT REAL
                if cmd == "FORWARD":
                    rob.moveWheels(SPEED_FWD, SPEED_FWD)
                    rob.wait(CMD_TIME_SHORT)
                    rob.stopMotors()
                elif cmd == "TURN_LEFT":
                    rob.moveWheels(-TURN_SPEED, TURN_SPEED)
                    rob.wait(CMD_TIME_SHORT)
                    rob.stopMotors()
                elif cmd == "TURN_RIGHT":
                    rob.moveWheels(TURN_SPEED, -TURN_SPEED)
                    rob.wait(CMD_TIME_SHORT)
                    rob.stopMotors()
                else:
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

                # 🔹 Mostrar también la cámara del móvil (lo que ve el robot)
                if annotated_phone is not None:
                    cv2.putText(
                        annotated_phone,
                        f"MODE: {mode}",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Camara Robobo + YOLO", annotated_phone)

            # ---------- 3) MODO PPO ----------
            else:  # mode == "PPO"
                if frame_phone is None:
                    # Si por lo que sea no tenemos frame, paramos y esperamos
                    rob.stopMotors()
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                # Estado ya calculado antes: phone_state (0..13)
                obs = np.array([phone_state], dtype=np.int64)
                action, _ = ppo_model.predict(obs, deterministic=True)

                print(f"[PPO] state={phone_state}, action={int(action)}")
                apply_ppo_action(rob, action)
                ppo_steps += 1

                # Mostrar cámara del Robobo con detecciones + estado/acción
                if annotated_phone is not None:
                    cv2.putText(
                        annotated_phone,
                        f"MODE: {mode} state:{phone_state} action:{int(action)}",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Camara Robobo + YOLO", annotated_phone)

                if ppo_steps >= MAX_PPO_STEPS:
                    print("MAX_PPO_STEPS alcanzado. Terminando.")
                    break

            # ---------- 4) TECLA ESC PARA SALIR ----------
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cleanup(rob, video, cap, exit_code=0)


if __name__ == "__main__":
    main()
