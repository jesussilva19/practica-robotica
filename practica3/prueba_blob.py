import cv2, time, sys
import numpy as np
from ultralytics import YOLO

# Robobo (sim + SDK)
from robobosim.RoboboSim import RoboboSim
from robobopy.Robobo import Robobo
from robobopy.utils.BlobColor import BlobColor

# ==== PPO (Práctica 1) ====
from stable_baselines3 import PPO

# -------------------- CONFIG --------------------
# Pose (teleop)
MODEL_PATH   = "yolo11n-pose.pt"  # o "yolov8n-pose.pt"
IMG_SIZE     = 640
POSE_CONF    = 0.5
WRIST_MARGIN = 20
TORSO_X_OFF  = 60

# Robobo sim
ROBOBO_IP = "localhost"

# Histéresis de conmutación por blob rojo
ENTER_AUTO_FRAMES = 4
EXIT_AUTO_FRAMES  = 6
BLOB_SIZE_MIN     = 2.0   # umbral "lo veo" (ajusta si hace falta)

# Velocidades base para teleop (discretas)
SPEED_FWD   = 30
SPEED_TURN  = 25
CMD_MS      = 300  # ms

# PPO model path
PPO_MODEL_PATH = "best_model"
USE_ONE_HOT    = True  # <- si tu PPO fue entrenado con one-hot de 14 estados

# Estados discretos (como en tu RoboboEnv)
PAN_POSITIONS = [0, 15, 30, 45, 60, 75, 90, -15, -30, -45, -60, -75, -90]
NUM_STATES    = 14  # 0..12 (cada pan) y 13 == no visible

# ------------------------------------------------

# Índices COCO (17 kp)
NOSE=0; L_SHOULDER=5; R_SHOULDER=6; L_WRIST=9; R_WRIST=10

def safe_np(a):
    if a is None: return None
    try: return a.cpu().numpy()
    except Exception: return np.asarray(a)

# --------- TELEOP (YOLO Pose) ----------
def get_main_person_keypoints(result):
    kps = getattr(result, "keypoints", None)
    if kps is None:
        return None
    xy   = safe_np(getattr(kps, "xy", None))
    conf = safe_np(getattr(kps, "conf", None))
    if xy is None or xy.ndim != 3 or xy.shape[-1] != 2: return None
    n, K, _ = xy.shape
    if n == 0 or K < 11: return None
    if conf is not None and conf.ndim == 2 and conf.shape[0] == n:
        mean_conf = np.nanmean(conf, axis=1)
        idx = 0 if np.all(np.isnan(mean_conf)) else int(np.nanargmax(mean_conf))
    else:
        idx = 0
    kp = xy[idx]
    if kp.shape[0] < (R_WRIST + 1): return None
    if not np.isfinite(kp[:(R_WRIST + 1)]).all(): return None
    if kp.shape[0] >= 17: kp = kp[:17]
    return kp

def gesture_from_keypoints(kp):
    try:
        if kp is None or kp.shape[0] <= R_WRIST:
            return "STOP"
        lx, ly = kp[L_WRIST]; rx, ry = kp[R_WRIST]
        lsh_x, lsh_y = kp[L_SHOULDER]; rsh_x, rsh_y = kp[R_SHOULDER]
        vals = np.array([lx, ly, rx, ry, lsh_x, lsh_y, rsh_x, rsh_y], dtype=float)
        if not np.isfinite(vals).all(): return "STOP"

        arm_right_up = ry < (rsh_y - WRIST_MARGIN)
        arm_left_up  = ly < (lsh_y - WRIST_MARGIN)

        torso_delta = rsh_x - lsh_x
        if arm_left_up and arm_right_up: return "FORWARD"
        if arm_right_up and not arm_left_up: return "TURN_RIGHT"
        if arm_left_up and not arm_right_up: return "TURN_LEFT"
        if torso_delta < -TORSO_X_OFF: return "TURN_LEFT"
        if torso_delta >  TORSO_X_OFF: return "TURN_RIGHT"
        return "STOP"
    except Exception:
        return "STOP"

# --------- Conexión / actuadores básicos ----------
def connect_robobo(ip):
    try:
        r = Robobo(ip); r.connect()
        s = RoboboSim(ip); s.connect()
        return r, s, True
    except Exception as e:
        print(f"[WARN] No se pudo conectar a Robobo/Sim ({ip}): {e}", file=sys.stderr)
        return None, None, False

def act_manual(rob, cmd, connected_flag):
    if not connected_flag or rob is None:
        return connected_flag
    try:
        if cmd == "FORWARD":
            rob.moveWheels(SPEED_FWD, SPEED_FWD); time.sleep(CMD_MS/1000); rob.stopMotors()
        elif cmd == "TURN_LEFT":
            rob.moveWheels(-SPEED_TURN, SPEED_TURN); time.sleep(CMD_MS/1000); rob.stopMotors()
        elif cmd == "TURN_RIGHT":
            rob.moveWheels(SPEED_TURN, -SPEED_TURN); time.sleep(CMD_MS/1000); rob.stopMotors()
        else:
            rob.stopMotors()
        return True
    except Exception as e:
        print(f"[WARN] Conexión Robobo perdida (manual '{cmd}'): {e}", file=sys.stderr)
        return False

# --------- Lectura de blob rojo (disparador AUTO) ----------
def read_red_blob(rob):
    """
    Devuelve dict {'size': float, 'posx': float} o None.
    """
    try:
        b = rob.readColorBlob(BlobColor.RED)
        # API habitual: b.size, b.posx (0..100?), etc.
        # Si tu SDK usa otro rango/atributos, ajusta aquí.
        if b is None:
            return None
        size = float(getattr(b, "size", 0.0))
        posx = float(getattr(b, "posx", 50.0))  # 0..100 (habitual en Robobo)
        return {"size": size, "posx": posx}
    except Exception:
        return None

# --------- Estado discreto (como tu RoboboEnv._get_state) ----------
def get_discrete_state_like_env(rob) -> int:
    """
    Barre PAN_POSITIONS y retorna:
      0..12 si ve blob en esa posición,
      13 si no visible en ninguna.
    Emula tu _get_state() para emparejar la política entrenada.
    """
    for i, ang in enumerate(PAN_POSITIONS):
        # movePanTo(angle, speed, block) -> bloqueante True para emular env
        rob.movePanTo(ang, 100, True)
        b = read_red_blob(rob)
        if b is not None and b["size"] > 0:
            return i
    return len(PAN_POSITIONS)  # 13 == no visible

def make_obs_from_state(state: int):
    """
    Crea la observación para el PPO desde el estado discreto.
    - Si entrenaste con one-hot: vector (14,) con 1 en 'state'.
    - Si entrenaste con entero: devuelve np.array([[state]], dtype=np.int64) (raro en SB3).
    """
    if USE_ONE_HOT:
        vec = np.zeros((NUM_STATES,), dtype=np.float32)
        state_clamped = int(np.clip(state, 0, NUM_STATES-1))
        vec[state_clamped] = 1.0
        return vec.reshape(1, -1)
    else:
        return np.array([[int(state)]], dtype=np.int64)

# --------- Actuar en AUTO según acción 0..5 (como en tu env.step) ----------
def act_auto_env_action(rob, action: int, connected_flag: bool):
    if not connected_flag or rob is None:
        return connected_flag
    try:
        if action == 0:      # Avanzar recto
            rob.moveWheelsByTime(5, 5, 2)
        elif action == 1:    # Girar izquierda (leve)
            rob.moveWheelsByTime(0, 5, 2)
        elif action == 2:    # Girar derecha (leve)
            rob.moveWheelsByTime(5, 0, 2)
        elif action == 3:    # Girar izquierda (fuerte)
            rob.moveWheelsByTime(0, 5, 4)
        elif action == 4:    # Girar derecha (fuerte)
            rob.moveWheelsByTime(5, 0, 4)
        elif action == 5:    # Giro 180°
            rob.moveWheelsByTime(10, -10, 3)
        return True
    except Exception as e:
        print(f"[WARN] Conexión Robobo perdida (auto act {action}): {e}", file=sys.stderr)
        return False

# ===================== MAIN =====================
def main():
    # Cargar YOLO pose para teleop
    pose_model = YOLO(MODEL_PATH)

    # Cargar PPO entrenado (Práctica 1)
    try:
        ppo_model = PPO.load(PPO_MODEL_PATH)
        print("✅ PPO cargado:", PPO_MODEL_PATH)
    except Exception as e:
        print(f"[WARN] No se pudo cargar '{PPO_MODEL_PATH}': {e}")
        ppo_model = None

    # Conectar Robobo + Sim
    rob, sim, connected = connect_robobo(ROBOBO_IP)

    # Webcam PC para gestos
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARN] No se pudo abrir la webcam; TELEOP no disponible.", file=sys.stderr)

    mode = "TELEOP"
    seen_cnt = 0
    miss_cnt = 0

    print("TELEOP por gestos; al ver BLOB ROJO conmutará a AUTO (PPO de Práctica 1). ESC para salir.")

    try:
        while True:
            # 1) Mirar blob rojo con Robobo (trigger de modo)
            blob = read_red_blob(rob) if connected else None
            sees_red = (blob is not None) and (blob["size"] >= BLOB_SIZE_MIN)

            if sees_red:
                seen_cnt += 1; miss_cnt = 0
            else:
                miss_cnt += 1; seen_cnt = 0

            if mode == "TELEOP" and seen_cnt >= ENTER_AUTO_FRAMES:
                mode = "AUTO"; print("[MODO] AUTO (PPO)")
            elif mode == "AUTO" and miss_cnt >= EXIT_AUTO_FRAMES:
                mode = "TELEOP"; print("[MODO] TELEOP (gestos)")

            # 2) Ejecutar control según modo
            hud = None
            cmd_text = "STOP"

            if mode == "TELEOP" and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                r = pose_model.predict(frame, imgsz=IMG_SIZE, conf=POSE_CONF, verbose=False)[0]
                kp = get_main_person_keypoints(r)
                cmd = gesture_from_keypoints(kp)
                connected = act_manual(rob, cmd, connected)
                hud = r.plot()
                cmd_text = cmd
            else:
                # AUTO: usar política PPO de la práctica 1 con el estado discreto
                # replicado de tu RoboboEnv._get_state()
                state = get_discrete_state_like_env(rob)
                obs = make_obs_from_state(state)

                if ppo_model is not None:
                    try:
                        action, _ = ppo_model.predict(obs, deterministic=True)
                        # SB3 suele devolver un np.ndarray; fuerza a int
                        if isinstance(action, (np.ndarray, list)):
                            action = int(action[0])
                        else:
                            action = int(action)
                    except Exception as e:
                        print(f"[WARN] PPO.predict falló: {e}")
                        action = 0  # STOP como salvaguarda
                else:
                    action = 0  # Si no cargó, nos quedamos quietos

                connected = act_auto_env_action(rob, action, connected)
                cmd_text = f"AUTO: act={action} state={state}"

                # En AUTO no tenemos imagen “bonita”, pintamos un lienzo simple
                hud = np.zeros((480, 640, 3), dtype=np.uint8)

            # 3) Reconexión suave cada ~3s si cae
            if not connected and (int(time.time()) % 3 == 0):
                rob, sim, connected = connect_robobo(ROBOBO_IP)

            # 4) HUD
            if hud is None:
                hud = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(hud, f"MODE: {mode}  CMD: {cmd_text}{'' if connected else ' (sin Robobo)'}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(hud, f"seen:{seen_cnt} miss:{miss_cnt} blob_sz:{(blob['size'] if blob else 0):.1f}",
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Teleop (pose) -> AUTO (PPO) por blob rojo", hud)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        try:
            if rob is not None:
                rob.stopMotors(); rob.disconnect()
            if sim is not None:
                sim.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    main()
