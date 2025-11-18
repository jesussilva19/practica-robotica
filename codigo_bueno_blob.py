"""
TELEOP (YOLO-Pose) -> AR (PPO) con cambio cuando se ve el blob rojo (blob.size > 0).
Un único episodio. Sin mapas ni estadísticas.
"""

import os
import time
import numpy as np
import cv2
from ultralytics import YOLO
from stable_baselines3 import PPO

from practica3.main_ppo import RoboboEnv
from robobopy.utils.BlobColor import BlobColor

# ========================= CONFIG =========================
BASE_DIR = os.path.dirname(__file__)
PPO_MODEL_PATH = os.path.join(BASE_DIR, 'practica3', 'best_model.zip')

MAX_STEPS = 150
RENDER = False

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
BLOB_SEEN_MIN_SIZE = 0.0  # cambia a >0 si quieres evitar falsos positivos mínimos

# ===================== TELEOP HELPERS =====================
# Índices COCO (17 kp)
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
def run_one_episode(ppo_model, max_steps=150, render=False):
    """
    - TELEOP con YOLO-Pose hasta ver blob rojo (size > 0).
    - Cambia a AR (PPO) sin resetear simulación/conexiones.
    """
    env = RoboboEnv(max_steps=max_steps)  # conecta robobo y sim internamente
    obs, _ = env.reset()

    # Preparar teleop
    cap = cv2.VideoCapture(0)
    pose_model = YOLO(POSE_MODEL_PATH)

    total_reward = 0.0
    steps = 0
    done = False
    mode = "TELEOP"

    print("\nIniciando episodio (TELEOP → AR)")
    print("Cambio a AR cuando blob rojo sea visible (size > 0).")

    try:
        while not done and steps < max_steps:
            # TELEOP
            if mode == "TELEOP":
                
                blob = env.robobo.readColorBlob(BlobColor.RED)
                sees_red = (blob is not None) and (float(getattr(blob, "size", 0.0)) > BLOB_SEEN_MIN_SIZE)
                if sees_red:
                    mode = "AR"
                    print(f"\nCambio TELEOP ➜ AR (blob.size={float(getattr(blob, 'size', 0.0)):.1f})")
                    try:
                        obs = env._get_state()  # observa sin reset
                    except Exception:
                        pass
                    print("Pausa 2s...")
                    time.sleep(2)
                    continue

                ok, frame = cap.read()
                if ok:
                    r = pose_model.predict(frame, imgsz=IMG_SIZE, conf=POSE_CONF, verbose=False)[0]
                    kp = get_main_person_keypoints(r)
                    cmd = gesture_from_keypoints(kp)
                else:
                    cmd = "STOP"

                # Ejecutar gesto
                if cmd == "FORWARD":
                    env.robobo.moveWheels(SPEED_FWD, SPEED_FWD); time.sleep(CMD_MS/1000); env.robobo.stopMotors()
                elif cmd == "TURN_LEFT":
                    env.robobo.moveWheels(-SPEED_TURN, SPEED_TURN); time.sleep(CMD_MS/1000); env.robobo.stopMotors()
                elif cmd == "TURN_RIGHT":
                    env.robobo.moveWheels(SPEED_TURN, -SPEED_TURN); time.sleep(CMD_MS/1000); env.robobo.stopMotors()
                else:
                    env.robobo.stopMotors()

                steps += 1
                if render and ok:
                    ann = r.plot()
                    cv2.putText(ann, f"MODE: {mode} CMD:{cmd}", (15, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.imshow("TELEOP (YOLO-Pose)", ann)
                    if cv2.waitKey(1) & 0xFF == 27: break

            # AR (PPO)
            else:
                obs = env._get_state()
                action, _ = ppo_model.predict(np.array([obs]) if np.isscalar(obs) else obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                total_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)
                if render:
                    env.render()

        # (opcional) pequeño resumen por consola
        try:
            success = done and env._is_at_goal()
        except Exception:
            success = False
        print(f"\nEpisodio terminado. Éxito: {success}. Recompensa AR acumulada: {total_reward:.2f}")

    finally:
        try:
            if cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            env.close()
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

    run_one_episode(ppo_model, max_steps=MAX_STEPS, render=RENDER)

if __name__ == "__main__":
    main()
