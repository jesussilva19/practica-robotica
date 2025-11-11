import numpy as np

# idx COCO (aprox.): 5=Shoulder_L, 6=Shoulder_R, 7=Elbow_L, 8=Elbow_R, 9=Wrist_L, 10=Wrist_R
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10

def get_main_person_keypoints(result):
    if result.keypoints is None or len(result.keypoints) == 0:
        return None  # nadie
    # coge la persona con mayor confianza/media conf.
    kps = result.keypoints
    confs = kps.conf.squeeze(-1).mean(dim=1).cpu().numpy()
    idx = int(np.argmax(confs))
    return kps[idx].xy.cpu().numpy()  # shape (17, 2) -> (x,y)

def gesture_from_keypoints(kp):
    # Reglas sencillas (tuneables)
    lx, ly = kp[L_WRIST]
    rx, ry = kp[R_WRIST]
    lsx, lsy = kp[L_SHOULDER]
    rsx, rsy = kp[R_SHOULDER]

    arm_right_up = ry < rsy - 20
    arm_left_up  = ly < lsy - 20

    # inclinación del torso: diferencia horizontal hombros
    torso_delta = (rsx - lsx)
    lean_right = torso_delta > 60
    lean_left  = torso_delta < -60

    if arm_left_up and arm_right_up:
        return "FORWARD"
    if arm_right_up and not arm_left_up:
        return "TURN_RIGHT"
    if arm_left_up and not arm_right_up:
        return "TURN_LEFT"
    if lean_left:
        return "TURN_LEFT"
    if lean_right:
        return "TURN_RIGHT"
    return "STOP"
