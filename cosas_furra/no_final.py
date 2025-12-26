from robobopy_videostream.RoboboVideo import RoboboVideo
from robobopy.Robobo import Robobo
from ultralytics import YOLO
import cv2
import signal
import sys
import time
import numpy as np

# ================== CONFIG ==================
IP = "172.20.10.2"          # IP del móvil con la app del Robobo (la misma que te funciona)
POSE_MODEL_PATH = "yolo11n-pose.pt"   # modelo pose
DET_MODEL_PATH  = "yolo11n.pt"        # modelo detección COCO

IMG_SIZE    = 640
POSE_CONF   = 0.5
MOUSE_CONF  = 0.8

SPEED_FWD   = 20
SPEED_TURN  = 15
CMD_SEC     = 0.4     # duración de cada comando de movimiento (segundos)

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

    print("Teleoperación lista.")
    print("Gestos: ambos brazos=adelante; brazo izq=izq; brazo dcho=dcha; sin gesto=stop.")
    print("La cámara del Robobo detectará 'mouse' con conf > 0.8. ESC para salir.\n")

    try:
        while True:
            # ---------- 1) LEER CÁMARA DEL MÓVIL Y BUSCAR MOUSE ----------
            annotated_phone = None
            try:
                frame_phone, ts, sync_id, frame_id = video.getImageWithMetadata()
            except Exception:
                frame_phone = None

            if frame_phone is not None:
                det_results = det_model.predict(frame_phone, imgsz=IMG_SIZE, conf=0.5, verbose=False)
                annotated_phone = det_results[0].plot()

                mouse_detected = False
                best_conf = 0.0

                for box in det_results[0].boxes:
                    cls_id = int(box.cls)
                    cls_name = det_results[0].names[cls_id]
                    conf = float(box.conf)

                    if cls_name == "cell phone" and conf > MOUSE_CONF:
                        mouse_detected = True
                        best_conf = conf
                        break

                if mouse_detected:
                    print(f"\nRATÓN DETECTADO EN CÁMARA DEL ROBOBO → conf={best_conf:.2f}")
                    print("Parando robot y saliendo...")
                    cleanup(rob, video, cap, exit_code=0)

            # ---------- 2) LEER WEBCAM PC Y CONTROLAR POR GESTOS ----------
            ok, frame_pc = cap.read()
            if not ok:
                # Aun así refrescamos ventanas/teclas
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            pose_results = pose_model.predict(frame_pc, imgsz=IMG_SIZE, conf=POSE_CONF, verbose=False)
            r_pose = pose_results[0]
            kp = get_main_person_keypoints(r_pose)
            cmd = gesture_from_keypoints(kp)

            print(f"CMD: {cmd}")

            # Ejecutar comando en el ROBOT REAL
            if cmd == "FORWARD":
                rob.moveWheels(SPEED_FWD, SPEED_FWD)
                rob.wait(CMD_SEC)
                rob.stopMotors()
            elif cmd == "TURN_LEFT":
                rob.moveWheels(-SPEED_TURN, SPEED_TURN)
                rob.wait(CMD_SEC)
                rob.stopMotors()
            elif cmd == "TURN_RIGHT":
                rob.moveWheels(SPEED_TURN, -SPEED_TURN)
                rob.wait(CMD_SEC)
                rob.stopMotors()
            else:
                rob.stopMotors()

            # ---------- 3) MOSTRAR VENTANAS ----------
            annotated_pc = r_pose.plot()
            cv2.putText(
                annotated_pc,
                f"CMD: {cmd}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Teleop Robobo REAL (YOLO-Pose)", annotated_pc)

            # Mostrar lo que ve el ROBOT (cámara del móvil)
            if annotated_phone is not None:
                cv2.imshow("Camara Robobo + YOLO", annotated_phone)

            # Una sola waitKey para ambas ventanas
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cleanup(rob, video, cap, exit_code=0)

if __name__ == "__main__":
    # Ctrl+C -> salida limpia
    def sig_handler(sig, frame):
        cleanup(rob=None, video=None, cap=None, exit_code=0)
    signal.signal(signal.SIGINT, sig_handler)

    main()
