import cv2, time, sys
from ultralytics import YOLO
import numpy as np
from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo

# -------------------- CONFIG --------------------
MODEL_PATH = "yolo11n-pose.pt"   # o "yolov8n-pose.pt" (debe ser *pose*)
ROBOBO_IP  = "localhost"      # cambia por la tuya

SPEED_FWD  = 30
SPEED_TURN = 25
CMD_MS     = 300                 # duración breve por comando (ms)
IMG_SIZE   = 640                 # baja a 480 si necesitas más FPS
CONF_TH    = 0.5                 # confianza mínima para pose
# ------------------------------------------------

# Índices COCO (17 kp)
NOSE=0; L_SHOULDER=5; R_SHOULDER=6; L_WRIST=9; R_WRIST=10

def safe_np(a):
    """Pasa tensores a numpy y garantiza np.ndarray."""
    if a is None:
        return None
    try:
        return a.cpu().numpy()
    except Exception:
        return np.asarray(a)

def get_main_person_keypoints(result):
    """
    Devuelve un np.ndarray (17,2) con (x,y) de la persona principal,
    o None si no hay pose válida o formato inesperado.
    """
    kps = getattr(result, "keypoints", None)
    if kps is None:
        return None

    xy   = safe_np(getattr(kps, "xy", None))     # shape esperada [n,17,2]
    conf = safe_np(getattr(kps, "conf", None))   # shape esperada [n,17]

    if xy is None or xy.ndim != 3 or xy.shape[-1] != 2:
        return None
    n, K, _ = xy.shape
    if n == 0 or K < 11:
        return None

    # Elegir persona por confianza media si existe, si no idx=0
    if conf is not None and conf.ndim == 2 and conf.shape[0] == n:
        # descarta personas con muchas NaN/inf
        mean_conf = np.nanmean(conf, axis=1)
        if np.all(np.isnan(mean_conf)):
            idx = 0
        else:
            idx = int(np.nanargmax(mean_conf))
    else:
        idx = 0

    kp = xy[idx]  # (K,2)
    # Validación de forma/finito al menos hasta R_WRIST
    if kp.shape[0] < (R_WRIST + 1):
        return None
    if not np.isfinite(kp[:(R_WRIST + 1)]).all():
        return None
    # Si el modelo da >17 puntos (algunas variantes), recortamos a 17
    if kp.shape[0] >= 17:
        kp = kp[:17]
    return kp  # (17,2)

def gesture_from_keypoints(kp):
    """
    Devuelve: 'FORWARD' | 'TURN_LEFT' | 'TURN_RIGHT' | 'STOP'
    Robusta a valores perdidos. Si falta info => STOP.
    """
    try:
        if kp is None or kp.shape[0] <= R_WRIST:
            return "STOP"

        lx, ly = kp[L_WRIST]
        rx, ry = kp[R_WRIST]
        lsh_x, lsh_y = kp[L_SHOULDER]
        rsh_x, rsh_y = kp[R_SHOULDER]

        # Comprobar finitud
        vals = np.array([lx, ly, rx, ry, lsh_x, lsh_y, rsh_x, rsh_y], dtype=float)
        if not np.isfinite(vals).all():
            return "STOP"

        # Reglas simples (tunea márgenes a tu altura/encuadre)
        arm_right_up = ry < (rsh_y - 20)
        arm_left_up  = ly < (lsh_y - 20)

        torso_delta = rsh_x - lsh_x  # >0: hombro dcho a la derecha del izq
        if arm_left_up and arm_right_up:
            return "FORWARD"
        if arm_right_up and not arm_left_up:
            return "TURN_RIGHT"
        if arm_left_up and not arm_right_up:
            return "TURN_LEFT"
        if torso_delta < -60:
            return "TURN_LEFT"
        if torso_delta > 60:
            return "TURN_RIGHT"
        return "STOP"
    except Exception:
        return "STOP"

def connect_robobo(ip):
    try:
        r = Robobo(ip)
        r.connect()
        return r, True
    except Exception as e:
        print(f"[WARN] No se pudo conectar a Robobo ({ip}): {e}", file=sys.stderr)
        return None, False

def act(rob, cmd, connected_flag):
    """
    Ejecuta comando en Robobo si hay conexión; si no, ignora sin romper el bucle.
    Devuelve el flag de conexión actualizado (si se pierde al enviar).
    """
    if not connected_flag or rob is None:
        return connected_flag
    try:
        if cmd == "FORWARD":
            rob.moveWheels(SPEED_FWD, SPEED_FWD)
            time.sleep(CMD_MS/1000)
            rob.stopMotors()
        elif cmd == "TURN_LEFT":
            rob.moveWheels(-SPEED_TURN, SPEED_TURN)
            time.sleep(CMD_MS/1000)
            rob.stopMotors()
        elif cmd == "TURN_RIGHT":
            rob.moveWheels(SPEED_TURN, -SPEED_TURN)
            time.sleep(CMD_MS/1000)
            rob.stopMotors()
        else:  # STOP
            rob.stopMotors()
        return True
    except Exception as e:
        print(f"[WARN] Conexión Robobo perdida durante '{cmd}': {e}", file=sys.stderr)
        return False

def main():
    # Modelo pose
    model = YOLO(MODEL_PATH)

    # Conexión inicial a Robobo (si falla, seguimos con visión/gestos)
    rob, connected = connect_robobo(ROBOBO_IP)

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "No se pudo abrir la cámara"

    print("Teleoperación lista. Gestos: ambos brazos=adelante; brazo izq=izq; brazo dcho=dcha; sin gesto=stop.")
    print("Pulsa ESC para salir.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_TH, verbose=False)
            r = results[0]

            kp = get_main_person_keypoints(r)
            cmd = gesture_from_keypoints(kp)

            # Intentar actuar; si se perdió conexión, no crashear
            connected = act(rob, cmd, connected)

            # Dibujo
            annotated = r.plot()
            cv2.putText(annotated, f"CMD: {cmd}{'' if connected else ' (sin Robobo)'}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("Robobo Teleop via YOLO-Pose", annotated)

            # Si perdimos conexión, intentamos reconectar de forma ocasional (no en cada frame)
            if not connected and (int(time.time()) % 3 == 0):  # cada ~3s
                rob, connected = connect_robobo(ROBOBO_IP)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            if rob is not None:
                rob.stopMotors()
                rob.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    main()
