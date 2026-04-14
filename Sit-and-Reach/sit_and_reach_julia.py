from pykinect2 import PyKinectRuntime, PyKinectV2
import mediapipe as mp
import datetime as dt
import pandas as pd
import numpy as np
import time
import math
import sys
import cv2

PIXEL_TO_CM_RATIO = 0.533333

MIN_POSTURE_ELBOW_ANGLE     = 155
MAX_POSTURE_ELBOW_ANGLE     = 180
MIN_CALIBRATION_ELBOW_ANGLE = 20
MAX_CALIBRATION_ELBOW_ANGLE = 120
MIN_OPPOSITE_ELBOW_ANGLE    = 140
MAX_OPPOSITE_ELBOW_ANGLE    = 180
MIN_KNEE_ANGLE              = 100
MAX_KNEE_ANGLE              = 180
MIN_CALIBRATION_HIP_ANGLE   = 120
MAX_CALIBRATION_HIP_ANGLE   = 160
MIN_OPPOSITE_KNEE_ANGLE     = 80
MAX_OPPOSITE_KNEE_ANGLE     = 150
MIN_POSTURE_HIP_ANGLE       = 55
MAX_POSTURE_HIP_ANGLE       = 150

ERROR                     = 1.035
CALIBRATION_HELD_DURATION = 3
POSE_HELD_DURATION        = 3
AVERAGE_OVER              = 6

C_PANEL   = (255,  255,  255)
C_ACCENT  = (0,  210, 255)
C_SUCCESS = (0,  220, 100)
C_WARN    = (0,  160, 255)
C_ERROR   = (60,  60, 200)
C_WHITE   = (255, 255, 255)
C_GREY    = (140, 140, 160)
C_YELLOW  = (0,  230, 230)

kinect            = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic       = mp.solutions.holistic
holistic          = mp_holistic.Holistic()


def draw_header(img, exercise_label, side_label, rep_num, total_reps):
    """Top bar: exercise name | side | rep counter."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 70), C_PANEL, -1)
    cv2.line(img, (0, 70), (w, 70), C_ACCENT, 2)
    cv2.putText(img, exercise_label, (20, 48),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, C_ACCENT, 2, cv2.LINE_AA)
    side_text = f"Side: {side_label}"
    (tw, _), _ = cv2.getTextSize(side_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(img, side_text, (w // 2 - tw // 2, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_YELLOW, 2, cv2.LINE_AA)
    rep_text = f"Rep {rep_num}/{total_reps}"
    (tw2, _), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(img, rep_text, (w - tw2 - 20, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_WHITE, 2, cv2.LINE_AA)


def draw_calibration_legend(img, calib_status):
    """
    Bottom-left panel — three rows for the three calibration states.
    Active state is highlighted; the others are dimmed.
    The original 'calibration' variable string is used directly.
    """
    h, w = img.shape[:2]
    px, py    = 20, h - 220
    panel_w   = 430
    panel_h   = 205

    overlay = img.copy()
    cv2.rectangle(overlay, (px - 10, py - 10),
                  (px + panel_w, py + panel_h), C_PANEL, -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, (px - 10, py - 10),
                  (px + panel_w, py + panel_h), C_ACCENT, 1)

    cv2.putText(img, "CALIBRATION STATUS", (px, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_GREY, 1, cv2.LINE_AA)
    cv2.line(img, (px, py + 28), (px + panel_w - 10, py + 28), C_GREY, 1)

    # These three labels match exactly what check_calibration() returns
    states = [
        ("Wrong Position", C_ERROR,   "Adjust your posture"),
        ("Right Position", C_WARN,    "Hold still to calibrate"),
        ("Ok",             C_SUCCESS, "Calibrated — start the exercise"),
    ]

    for i, (label, col, hint) in enumerate(states):
        row_y  = py + 58 + i * 52
        active = (calib_status == label)

        sq_col = col if active else (50, 50, 70)
        cv2.rectangle(img, (px, row_y - 18), (px + 16, row_y + 6), sq_col, -1)
        if active:
            cv2.rectangle(img, (px, row_y - 18), (px + 16, row_y + 6), C_WHITE, 1)

        txt_col   = C_WHITE if active else C_GREY
        font_sz   = 0.70    if active else 0.60
        thickness = 2       if active else 1
        cv2.putText(img, label, (px + 26, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_sz, txt_col, thickness, cv2.LINE_AA)

        hint_col = col if active else (65, 65, 85)
        cv2.putText(img, hint, (px + 26, row_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, hint_col, 1, cv2.LINE_AA)

        if active:
            cv2.putText(img, "<<", (px + panel_w - 40, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    return py + panel_h   # bottom y for distance text

def _gradient_bg(h=500, w=900):
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    return img


def _base_screen(title_text, lines, prompt):
    img = _gradient_bg()
    cv2.rectangle(img, (0, 0), (900, 6), C_ACCENT, -1)
    (tw, _), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2)
    cv2.putText(img, title_text, (900 // 2 - tw // 2, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, (60, 100), (840, 100), C_ACCENT, 1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (60, 155 + i * 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, (60, 420), (840, 420), C_GREY, 1)
    (pw, _), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
    cv2.putText(img, prompt, (900 // 2 - pw // 2, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def finish_program():
    cv2.destroyAllWindows()
    kinect.close()
    sys.exit(0)

def calculate_distance_2d(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def average_distance(distances):
    return sum(distances) / len(distances)

def draw_dynamic_angle_arc(image, p1, p2, p3, angle):
    cv2.line(image, p1, p2, (255, 255, 0), 2)
    cv2.line(image, p2, p3, (255, 255, 0), 2)
    p1 = np.array(p1); p2 = np.array(p2); p3 = np.array(p3)
    arc_center = tuple(p2.astype(int))
    radius = int(np.linalg.norm(p1 - p2) / 2)
    axes = (radius, radius)
    cv2.ellipse(image, arc_center, axes, 0, 0, angle, (0, 255, 0), -1)
    cv2.ellipse(image, arc_center, axes, 0, 0, angle, (0, 0, 255), 2)

def calculate_angle(a, b, c):
    vetor_1 = (a[0] - b[0], a[1] - b[1])
    vetor_2 = (c[0] - b[0], c[1] - b[1])
    produto_escalar = vetor_1[0] * vetor_2[0] + vetor_1[1] * vetor_2[1]
    norma_vetor_1 = math.sqrt(vetor_1[0]**2 + vetor_1[1]**2)
    norma_vetor_2 = math.sqrt(vetor_2[0]**2 + vetor_2[1]**2)
    cos_angulo = produto_escalar / (norma_vetor_1 * norma_vetor_2)
    cos_angulo = max(-1, min(1, cos_angulo))
    angulo_radianos = math.acos(cos_angulo)
    return math.degrees(angulo_radianos)

def process_frame(kinect):
    frame = kinect.get_last_color_frame()
    frame = frame.reshape((1080, 1920, 4))
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    rgb_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    rgb_frame.flags.writeable = True
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), holistic.process(rgb_frame), frame

def process_landmarks(results, repeats):
    pose_landmarks, hand_landmarks = get_landmarks(results, repeats)
    if pose_landmarks is None or hand_landmarks is None:
        return None, None
    side = "right" if repeats in [0, 1] else "left"
    pose_indices = {
        "right": [15, 19, 29, 23, 25, 27, 11, 13, 26],
        "left":  [16, 20, 30, 24, 26, 28, 12, 14, 25]
    }
    required_pose_landmarks = [pose_landmarks[i] for i in pose_indices[side]]
    if all(lm.visibility > 0.0 for lm in required_pose_landmarks):
        return pose_landmarks, hand_landmarks
    return None, None

def draw_landmarks(image, results, repeats):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    if repeats in [0, 1]:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    else:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

def get_landmarks(results, repeats):
    if repeats in [0, 1]:
        if not results.pose_landmarks or not results.left_hand_landmarks:
            return None, None
        verification_hand, hand_landmarks = results.left_hand_landmarks, results.left_hand_landmarks.landmark
    else:
        if not results.pose_landmarks or not results.right_hand_landmarks:
            return None, None
        verification_hand, hand_landmarks = results.right_hand_landmarks, results.right_hand_landmarks.landmark
    pose_landmarks = results.pose_landmarks.landmark
    if results.pose_landmarks and verification_hand:
        return pose_landmarks, hand_landmarks
    return None, None

def draw_angles_arcs(repeats, knee_angle, opposite_knee_angle, hip_angle,
                     elbow_angle, opposite_elbow_angle, pose_landmarks, image, frame):
    side = "right" if repeats in [0, 1] else "left"
    pose_indices = {
        "right": [11, 13, 15, 23, 25, 27, 12, 14, 16],
        "left":  [12, 14, 16, 24, 26, 28, 11, 13, 15]
    }
    indices = pose_indices[side]
    def lm(i): return np.array([pose_landmarks[indices[i]].x, pose_landmarks[indices[i]].y])
    shoulder = lm(0); elbow = lm(1); wrist = lm(2)
    hip = lm(3);      knee  = lm(4); ankle = lm(5)
    o_shoulder = lm(6); o_elbow = lm(7); o_wrist = lm(8)
    def coord(v): return tuple(np.multiply(v[:2], [frame.shape[1], frame.shape[0]]).astype(int))
    shoulder_c = coord(shoulder); elbow_c   = coord(elbow);   wrist_c  = coord(wrist)
    hip_c      = coord(hip);      knee_c    = coord(knee);    ankle_c  = coord(ankle)
    o_shoulder_c = coord(o_shoulder); o_elbow_c = coord(o_elbow); o_wrist_c = coord(o_wrist)
    cv2.putText(image, f'Opposite Knee Angle: {opposite_knee_angle:.2f}',
                (1000, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
    cv2.putText(image, f'Opposite Elbow Angle: {opposite_elbow_angle:.2f}',
                o_elbow_c, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
    draw_dynamic_angle_arc(image, o_shoulder_c, o_elbow_c, o_wrist_c, opposite_elbow_angle)
    draw_dynamic_angle_arc(image, hip_c, knee_c, ankle_c, knee_angle)
    cv2.putText(image, f'Knee Angle: {knee_angle:.2f}',
                knee_c, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 230, 0), 2)
    draw_dynamic_angle_arc(image, shoulder_c, hip_c, knee_c, hip_angle)
    cv2.putText(image, f'Hip Angle: {hip_angle:.2f}',
                hip_c, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
    draw_dynamic_angle_arc(image, shoulder_c, elbow_c, wrist_c, elbow_angle)
    cv2.putText(image, f'Elbow Angle: {elbow_angle:.2f}',
                elbow_c, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

def calculate_angles(repeats, pose_landmarks):
    side = "right" if repeats in [0, 1] else "left"
    pose_indices = {
        "right": [11, 13, 15, 23, 25, 27, 24, 26, 28, 12, 14, 16],
        "left":  [12, 14, 16, 24, 26, 28, 23, 25, 27, 11, 13, 15]
    }
    indices = pose_indices[side]
    def pt(i): return np.array([pose_landmarks[indices[i]].x, pose_landmarks[indices[i]].y])
    shoulder = pt(0); elbow = pt(1); wrist = pt(2)
    hip = pt(3);      knee  = pt(4); ankle = pt(5)
    o_hip = pt(6);    o_knee = pt(7); o_ankle = pt(8)
    o_shoulder = pt(9); o_elbow = pt(10); o_wrist = pt(11)
    return (calculate_angle(hip, knee, ankle),
            calculate_angle(o_hip, o_knee, o_ankle),
            calculate_angle(shoulder, hip, knee),
            calculate_angle(shoulder, elbow, wrist),
            calculate_angle(o_shoulder, o_elbow, o_wrist))

def check_calibration(calibration_time, foot, repeats, knee_angle, opposite_knee_angle,
                      hip_angle, elbow_angle, progress_calibration, progress_calibration1,
                      calibration_held_duration, pose_landmarks):
    if repeats in [0, 1]:
        foot_index = 31
    else:
        foot_index = 32
    if (MIN_KNEE_ANGLE < knee_angle < MAX_KNEE_ANGLE and
            MIN_CALIBRATION_HIP_ANGLE < hip_angle < MAX_CALIBRATION_HIP_ANGLE and
            MIN_CALIBRATION_ELBOW_ANGLE < elbow_angle < MAX_CALIBRATION_ELBOW_ANGLE and
            progress_calibration1 == 0.0):
        if calibration_time is None:
            calibration_time = time.time()
        progress_calibration = (time.time() - calibration_time) / calibration_held_duration
        if progress_calibration >= 1.0:
            foot_landmark = pose_landmarks[foot_index]
            foot = int(foot_landmark.x * 640), int(foot_landmark.y * 480)
            return "Ok", 1.0, calibration_time, foot, 1.0
        return "Right Position", progress_calibration, calibration_time, None, 0.0
    if progress_calibration1 == 0.0:
        return "Wrong Position", 0.0, None, None, 0.0
    return "Ok", 1.0, calibration_time, foot, 1.0

def check_posture(pose_correct_start_time, knee_angle, opposite_knee_angle, hip_angle,
                  elbow_angle, opposite_elbow_angle, pose_held_duration, progress, distance):
    if (MIN_POSTURE_ELBOW_ANGLE < elbow_angle < MAX_POSTURE_ELBOW_ANGLE and
            MIN_OPPOSITE_ELBOW_ANGLE < opposite_elbow_angle < MAX_OPPOSITE_ELBOW_ANGLE and
            MIN_POSTURE_HIP_ANGLE < hip_angle < MAX_POSTURE_HIP_ANGLE and
            MIN_KNEE_ANGLE < knee_angle < MAX_KNEE_ANGLE):
        if pose_correct_start_time is None:
            pose_correct_start_time = time.time()
        progress = (time.time() - pose_correct_start_time) / pose_held_duration
        if progress >= 1.0:
            return "Correct", min(progress, 1.0), pose_correct_start_time, -distance
        return "Correct", min(progress, 1.0), pose_correct_start_time, None
    return "Incorrect", 0.0, None, None


# ═════════════════════════════════════════════════════════════════
#  RESULT / INPUT SCREENS  (styled v2 versions)
# ═════════════════════════════════════════════════════════════════

def final_visualization(left, right):
    lines = [
        (f"Best result of the right leg: {right} cm", C_SUCCESS),
        (f"Best result of the left leg : {left}  cm", C_SUCCESS),
    ]
    img = _base_screen("Exercise Completed", lines, 'Press  "Q"  to finish')
    cv2.imshow("Final Results", img)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def final_repetition_visualization(final_distance, real_distance,
                                   exercise_label, side_label, rep_num):
    lines = [
        (f"Final Distance : {final_distance} cm",     C_SUCCESS),
        (f"Real Distance  : {real_distance} cm",      C_ACCENT),
        (f"{exercise_label}  |  Side: {side_label}  |  Rep {rep_num}", C_GREY),
    ]
    img = _base_screen("Repetition Completed", lines,
                       'Press  "C"  to continue  |  "Q"  to quit')
    cv2.imshow("Final Repetition Results", img)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.destroyWindow("Final Repetition Results"); break
        elif key == ord('q'):
            finish_program()

def register():
    fields = ["Idade", "Altura (cm)", "Peso (kg)", "Genero (M/F)"]
    values = ["", "", "", ""]
    active_field = -1
    positions = [(50, 50 + i * 80, 550, 100 + i * 80) for i in range(len(fields))]

    def mouse_callback(event, x, y, flags, param):
        nonlocal active_field
        if event == cv2.EVENT_LBUTTONDOWN:
            active_field = -1
            for i, (x1, y1, x2, y2) in enumerate(positions):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    active_field = i; break

    cv2.namedWindow("Cadastro")
    cv2.setMouseCallback("Cadastro", mouse_callback)
    while True:
        img = 255 * np.ones((400, 600, 3), dtype=np.uint8)
        for i, (x1, y1, x2, y2) in enumerate(positions):
            background_color = (230, 230, 230)
            cv2.rectangle(img, (x1, y1), (x2, y2), background_color, -1)
            border_color = (0, 255, 0) if i == active_field else (0, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2)
            cv2.putText(img, f"{fields[i]}:", (x1 + 10, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(img, values[i], (x1 + 10, y2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, "Aperte Enter para finalizar", (50, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        cv2.imshow("Cadastro", img)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:   finish_program()
        elif key in (13, 10): cv2.destroyAllWindows(); return values
        elif key == 9:  active_field = (active_field + 1) % len(fields)
        elif active_field != -1:
            if key == 8: values[active_field] = values[active_field][:-1]
            elif 32 <= key <= 126: values[active_field] += chr(key)

def real_distance():
    distancia = ""
    cv2.namedWindow("Real Distance")
    while True:
        img = np.ones((200, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 60), (550, 120), (230, 230, 230), -1)
        cv2.rectangle(img, (50, 60), (550, 120), (0, 0, 0), 2)
        cv2.putText(img, "Digite a Distância medida (cm):", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, distancia, (60, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(img, "Pressione Enter para confirmar", (50, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        cv2.imshow("Real Distance", img)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:   cv2.destroyAllWindows(); finish_program()
        elif key in (13, 10):
            if distancia: cv2.destroyAllWindows(); return float(distancia.replace(",", "."))
        elif key == 8:  distancia = distancia[:-1]
        elif (key >= 48 and key <= 57) or key in [44, 46, 43, 45]: distancia += chr(key)


# ═════════════════════════════════════════════════════════════════
#  MAIN EXERCISE LOOP  — original logic + v2 overlays on top
# ═════════════════════════════════════════════════════════════════

def process_exercise(repeats):
    side_label = "Right" if repeats in [0, 1] else "Left"
    rep_num    = (repeats % 2) + 1

    # ── original state variables (unchanged) ──
    pose_correct_start_time = None
    calibration             = "Wrong Position"
    progress_calibration1   = 0
    progress_calibration    = 0
    calibration_time        = None
    final_distance          = None
    distances               = []
    foot                    = None

    while True:
        if kinect.has_new_color_frame():
            image, results, frame = process_frame(kinect)

            pose_correct = "Incorrect"
            progress     = 0

            pose_landmarks, hand_landmarks = process_landmarks(results, repeats)

            if pose_landmarks is not None and hand_landmarks is not None:

                draw_landmarks(image, results, repeats)

                angles = calculate_angles(repeats, pose_landmarks)
                draw_angles_arcs(repeats, *angles, pose_landmarks, image, frame)

                # ── original calibration call (unchanged) ──
                calibration, progress_calibration, calibration_time, foot, progress_calibration1 = \
                    check_calibration(calibration_time, foot, repeats, *angles[:4],
                                      progress_calibration, progress_calibration1,
                                      CALIBRATION_HELD_DURATION, pose_landmarks)

                if calibration == "Ok":
                    hand_landmark = hand_landmarks[12]
                    if repeats in [0, 1]:
                        hand = int((hand_landmark.x * 640) + 5), int((hand_landmark.y * 480) + 8)
                    else:
                        hand = int((hand_landmark.x * 640) - 3), int((hand_landmark.y * 480) + 13)

                    dist_pixels = calculate_distance_2d(hand, foot)
                    distance    = dist_pixels * PIXEL_TO_CM_RATIO

                    # original coord display (kept as-is)
                    cv2.putText(image, f'Position X and Y of foot: {foot[0]}, {foot[1]}',
                                (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
                    cv2.putText(image, f'Position X and Y of hand: {hand[0]}, {hand[1]}',
                                (1000, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

                    distances.append(distance)
                    if len(distances) > AVERAGE_OVER:
                        distances.pop(0)
                        distance = average_distance(distances)

                    pose_correct, progress, pose_correct_start_time, final_distance = \
                        check_posture(pose_correct_start_time, *angles,
                                      POSE_HELD_DURATION, progress, distance)

                    if final_distance != None:
                        if repeats in [0, 1]:
                            if hand[0] < foot[0] and distance > 1.2:
                                final_distance = -(final_distance + ERROR)
                        else:
                            if hand[0] > foot[0] and distance > 1.2:
                                final_distance = -(final_distance + ERROR)
                        break

                    # original distance / pose text (kept as-is)
                    cv2.putText(image, f"Dist: {distance:.2f} cm",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(image, f'Pose: {pose_correct}',
                                (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 2)

            # ── v2 overlays (added on top, do not replace anything) ──
            draw_header(image, "Sit and Reach", side_label, rep_num, 2)
            draw_calibration_legend(image, calibration)

            # original calibration text still present (belt-and-suspenders)
            cv2.putText(image, f'Calibration: {calibration}',
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 2)

            cv2.imshow('MediaPipe Holistic', image)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                finish_program()

    return f'{final_distance:.2f}'

distances_right = []
distances_left  = []

repeats = 0
age, height, weight, gender = register()
gender = "Feminine" if gender == "F" else "Male"

while repeats < 4:
    side_label = "Right" if repeats in [0, 1] else "Left"
    rep_num    = (repeats % 2) + 1

    final_distance = process_exercise(repeats)

    if final_distance is not None:
        caminho_arquivo = "./tabelas_utentes/sit_and_reach_2_utentes.xlsx"
        df = pd.read_excel(caminho_arquivo, engine="openpyxl")
        real = real_distance()

        erro = np.abs(np.abs(float(real)) - np.abs(float(final_distance)))

        new_line = {
            "Age": age, "Height": height, "Weight": weight, "Gender": gender,
            "Real distance": real, "Calculated distance": final_distance, "Erro": erro
        }
        df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
        df.to_excel(caminho_arquivo, index=False, engine="openpyxl")

        if repeats in [0, 1]:
            distances_right.append(final_distance)
            side = "right"
        else:
            distances_left.append(final_distance)
            side = "left"

        with open("./logs_utentes/logs_sit_and_reach_utentes", "a") as arquivo:
            arquivo.write(f"{dt.datetime.now()}, {age}, {height}, {weight}, "
                          f"{gender}, {real}, {final_distance}, {side}\n")

        final_repetition_visualization(final_distance, real, "Sit and Reach", side_label, rep_num)
        repeats += 1
    else:
        print("Exercise not performed correctly")
        finish_program()

best_left = min(distances_left, key=lambda x: abs(float(x)))
best_right = min(distances_right, key=lambda x: abs(float(x)))

# emit results for runner BEFORE opening final window
print(f"SAR_RIGHT={best_right}")
print(f"SAR_LEFT={best_left}")
sys.stdout.flush()

final_visualization(best_left, best_right)