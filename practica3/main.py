import cv2, time
from ultralytics import YOLO
import numpy as np
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

MODEL_PATH = "yolo11n-pose.pt"   # o "yolov8n-pose.pt"
ROBOBO_IP  = "localhost"      # cambia por la tuya

model = YOLO(MODEL_PATH)
rob = Robobo(ROBOBO_IP)
rob.connect()

SPEED_FWD = 30
SPEED_TURN = 25
CMD_MS = 300   # duración breve por comando (ms)

def act(cmd):
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

def get_main_person_keypoints(result):
    if result.keypoints is None or len(result.keypoints) == 0:
        return None
    kps = result.keypoints
    confs = kps.conf.squeeze(-1).mean(dim=1).cpu().numpy()
    idx = int(np.argmax(confs))
    return kps[idx].xy.cpu().numpy()

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10

def gesture_from_keypoints(kp):
    lx, ly = kp[L_WRIST]
    rx, ry = kp[R_WRIST]
    _, lsy = kp[L_SHOULDER]
    _, rsy = kp[R_SHOULDER]

    arm_right_up = ry < rsy - 20
    arm_left_up  = ly < lsy - 20

    rsx, _ = kp[R_SHOULDER]
    lsx, _ = kp[L_SHOULDER]
    torso_delta = (rsx - lsx)

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

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "No se pudo abrir la cámara"

print("Teleoperación lista. Gestos: ambos brazos=adelante; brazo izq=izq; brazo dcho=dcha; sin gesto=stop.")

try:
    while True:
        ok, frame = cap.read()
        if not ok: break

        results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
        r = results[0]
        kp = get_main_person_keypoints(r)

        cmd = "STOP"
        if kp is not None:
            cmd = gesture_from_keypoints(kp)
            act(cmd)

        annotated = r.plot()
        cv2.putText(annotated, f"CMD: {cmd}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow("Robobo Teleop via YOLO-Pose", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    rob.stopMotors()
    rob.disconnect()
