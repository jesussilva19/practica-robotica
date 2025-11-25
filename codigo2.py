"""
Híbrido TELEOP (YOLO en webcam del PC) -> AR (PPO)
- TELEOP: control por gestos hasta que el Robobo vea el blob rojo (blob.size > 0)
- AR: entonces se crea RoboboEnv (PPO) reutilizando las conexiones existentes.
"""

import os
import time
import numpy as np

import cv2
from ultralytics import YOLO
from stable_baselines3 import PPO

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor

from practica3.main_ppo import RoboboEnv   # ajusta el import si va en p23.main_ppo

# ========================= CONFIG =========================
ROBOBO_IP = "172.20.10.2"    # sim
BASE_DIR = os.path.dirname(__file__)
PPO_MODEL_PATH = os.path.join(BASE_DIR, 'practica3', 'best_model.zip')

MAX_STEPS = 150

# YOLO-Pose (teleop)
POSE_MODEL_PATH = "yolo11n-pose.pt"   # o "yolov8n-pose.pt"
IMG_SIZE = 640
POSE_CONF = 0.5
WRIST_MARGIN = 20
TORSO_X_OFFSET = 60

# Actuación teleop (discreta)
SPEED_FWD = 30
SPEED_TURN = 25
CMD_MS = 300  # ms

# Disparador AR
BLOB_SEEN_MIN_SIZE = 30 # >0 vale

# ===================== TELEOP HELPERS =====================
NOSE=0; L_SHOULDER=5; R_SHOULDER=6; L_WRIST=9; R_WRIST=10

def get_main_person_keypoints(result):
    kps = getattr(result, "keypoints", None)
    if kps is None: return None
    xy = kps.xy
    try: xy = xy.cpu().numpy()
    except Exception: xy = np.asarray(xy)
    if xy.ndim != 3 or xy.shape[-1] != 2 or xy.shape[1] < 11 or xy.shape[0] == 0:
        return None
    idx = 0
    conf = getattr(kps, "conf", None)
    if conf is not None:
        c = conf
        try: c = c.cpu().numpy()
        except Exception: c = np.asarray(c)
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
        if kp is None or kp.shape[0] <= R_WRIST: return "STOP"
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
        if torso_delta < -TORSO_X_OFFSET: return "TURN_LEFT"
        if torso_delta >  TORSO_X_OFFSET: return "TURN_RIGHT"
        return "STOP"
    except Exception:
        return "STOP"

# ===================== EPISODIO ÚNICO =====================
def run_one_episode(ppo_model, max_steps=150):
    """
    - TELEOP con YOLO (webcam PC) controlando Robobo+Sim directamente.
    - Cuando blob rojo (rob.readColorBlob(RED)) tiene size > 0:
        * se crea RoboboEnv (PPO)
        * se le conectan las mismas instancias de robobo y sim
        * se pasa a modo AR (PPO) hasta terminar.
    """
    # 1) Conectar Robobo y Sim a mano (sin RoboboEnv todavía)
    rob = Robobo(ROBOBO_IP)
    rob.connect()

    sim = RoboboSim(ROBOBO_IP)
    sim.connect()

    # Config cámara de blobs (como en tus prácticas)
    rob.setActiveBlobs(red=True, green=False, blue=False, custom=False)

    # 2) Preparar teleop: webcam + YOLO pose
    cap = cv2.VideoCapture(0)
    pose_model = YOLO(POSE_MODEL_PATH)

    total_reward = 0.0
    steps = 0
    mode = "TELEOP"
    done = False

    env = None  # se creará en cuanto entremos en PPO

    print("\nIniciando episodio (TELEOP → PPO)")
    print("PPO solo se activará cuando el blob rojo sea visible (blob.size > 0).")
    print("Teleop: ambos brazos=adelante, brazo izq=izq, brazo dcho=dcha, sin gesto=stop.")

    try:
        while not done and steps < max_steps:
            # --- TELEOP MODE ---
            if mode == "TELEOP":
                # 1) comprobar blob rojo
                blob = rob.readColorBlob(BlobColor.RED)
                size = float(getattr(blob, "size", 0.0)) if blob is not None else 0.0
                if size > BLOB_SEEN_MIN_SIZE:
                    print(f"\nBlob rojo detectado (size={size:.1f}) -> cambiando a modo PPO")
                    mode = "PPO"

                    # Crear env PPO reutilizando conexiones existentes
                    env = RoboboEnv(max_steps=max_steps-steps, host=ROBOBO_IP)
                    # Sobrescribir sus robobo y sim para NO reconectar ni resetear
                    env.robobo = rob
                    env.sim = sim
                    env.steps = 0  # contador interno del env

                    # Obtener estado actual sin resetear simulación
                    try:
                        obs = env._get_state()
                    except Exception:
                        obs = None

                    print("Pausa 2s antes de empezar con PPO...")
                    time.sleep(2)
                    continue  # saltar a siguiente iteración ya en modo PPO

                # 2) si no hay blob, seguimos con gestos
                ok, frame = cap.read()
                cmd = "STOP"
                r = None
                if ok:
                    results = pose_model.predict(frame, imgsz=IMG_SIZE, conf=POSE_CONF, verbose=False)
                    r = results[0]
                    kp = get_main_person_keypoints(r)
                    cmd = gesture_from_keypoints(kp)

                # 3) actuar sobre el Robobo (no sobre env)
                if cmd == "FORWARD":
                    rob.moveWheels(SPEED_FWD, SPEED_FWD); time.sleep(CMD_MS/1000); rob.stopMotors()
                elif cmd == "TURN_LEFT":
                    rob.moveWheels(-SPEED_TURN, SPEED_TURN); time.sleep(CMD_MS/1000); rob.stopMotors()
                elif cmd == "TURN_RIGHT":
                    rob.moveWheels(SPEED_TURN, -SPEED_TURN); time.sleep(CMD_MS/1000); rob.stopMotors()
                else:
                    rob.stopMotors()

                steps += 1

                # Mostrar SIEMPRE la ventana de YOLO en modo TELEOP
                if ok and r is not None:
                    annotated = r.plot()
                    cv2.putText(annotated, f"MODE: {mode}  CMD: {cmd}",
                                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.imshow("TELEOP (YOLO-Pose)", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                else:
                    # aunque no haya frame válido, necesitamos waitKey para refrescar
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            # --- PPO MODE ---
            else:
                # aquí SÍ usamos el entorno y PPO
                obs = env._get_state()
                action, _ = ppo_model.predict(
                    np.array([obs]) if np.isscalar(obs) else obs,
                    deterministic=True
                )
                obs, reward, terminated, truncated, _ = env.step(int(action))
                total_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)

                # si quieres, puedes mostrar algo por consola
                # print(f"[PPO] step={steps}, action={action}, reward={reward}")

        print(f"\nEpisodio terminado. Recompensa total (PPO): {total_reward:.2f}")

    finally:
        try:
            if cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if env is not None:
                env.close()     # dentro debería desconectar robobo y sim
            else:
                rob.disconnect()
                sim.disconnect()
        except Exception:
            pass

# ========================= MAIN =========================
def main():
    try:
        ppo_model = PPO.load(PPO_MODEL_PATH)
        print(f"Modelo PPO cargado desde {PPO_MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo PPO: {e}")
        return

    run_one_episode(ppo_model, max_steps=MAX_STEPS)

if __name__ == "__main__":
    main()
