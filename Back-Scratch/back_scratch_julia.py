from pykinect2 import PyKinectRuntime, PyKinectV2
import mediapipe as mp
import datetime as dt
import pandas as pd
import numpy as np
import time
import sys
import cv2
import gettext
from PIL import Image, ImageDraw, ImageFont

fontFile = "LiberationSansBold.ttf"

language = "en_US"
args = sys.argv
if len(args) >= 2:
    language = args[1]
lang = gettext.translation("messages", localedir="locale", languages=[language])
lang.install()

PIXEL_TO_CM_RATIO     = 0.625
POSE_NO_HELD_DURATION = 1.5
POSE_HELD_DURATION    = 3
DISTANCE_CHECK        = 33
AVERAGE_OVER          = 5
ERROR                 = 1.91

# ─────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (BGR)
# ─────────────────────────────────────────────────────────────────
C_PANEL   = (30,  30,  48)
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
    Bottom-left panel — three rows for the three Back Scratch states.
    States match exactly what the detection logic produces:
      'Not Detected' | 'Detected' | 'Ok'
    """
    h, w = img.shape[:2]
    px, py  = 20, h - 220
    panel_w = 430
    panel_h = 205

    overlay = img.copy()
    cv2.rectangle(overlay, (px - 10, py - 10),
                  (px + panel_w, py + panel_h), C_PANEL, -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, (px - 10, py - 10),
                  (px + panel_w, py + panel_h), C_ACCENT, 1)

    cv2.putText(img, "CALIBRATION STATUS", (px, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_GREY, 1, cv2.LINE_AA)
    cv2.line(img, (px, py + 28), (px + panel_w - 10, py + 28), C_GREY, 1)

    states = [
        ("Not Detected", C_ERROR,   "Position both hands in view"),
        ("Detected",     C_WARN,    "Move hands closer together"),
        ("Ok",           C_SUCCESS, "Hold position to measure"),
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

def process_frame(kinect):
    frame     = kinect.get_last_color_frame()
    frame     = frame.reshape((1080, 1920, 4))
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    rgb_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results   = holistic.process(rgb_frame)
    rgb_frame.flags.writeable = True
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), results, frame

def final_repetition_visualization(distance, real_distance,
                                   exercise_label, side_label, rep_num):
    lines = [
        (f"Distance between both hands: {distance} cm", C_SUCCESS),
        (f"Real Distance              : {real_distance} cm", C_ACCENT),
        (f"{exercise_label}  |  Side: {side_label}  |  Rep {rep_num}", C_GREY),
    ]
    img = _base_screen("Repetition Completed", lines,
                       'Press  "C"  to continue  |  "Q"  to quit')
    cv2.imshow("Repetition Results", img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): finish_program()
        elif key == ord('c'): cv2.destroyWindow("Repetition Results"); break

def final_visualization(left, right):
    lines = [
        (f"Best result of the right side: {right} cm", C_SUCCESS),
        (f"Best result of the left side : {left}  cm", C_SUCCESS),
    ]
    img = _base_screen("Exercise Completed", lines, 'Press  "Q"  to finish')
    cv2.imshow("Final Results", img)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

def check_distance(distance, start_time):
    if distance < DISTANCE_CHECK:
        if start_time is None:
            start_time = time.time()
        elapsed_time = time.time() - start_time
    else:
        start_time   = None
        elapsed_time = 0
    return elapsed_time, start_time

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
            cv2.rectangle(img, (x1, y1), (x2, y2), (230, 230, 230), -1)
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
#  MAIN EXERCISE LOOP — original logic + v2 overlays on top
# ═════════════════════════════════════════════════════════════════

def _hands_calib_state(results, distance):
    """Map detection state to one of the three legend labels."""
    if not (results.left_hand_landmarks and results.right_hand_landmarks):
        return "Not Detected"
    if distance is None or distance >= DISTANCE_CHECK:
        return "Detected"
    return "Ok"


def process_exercise(repeats):
    side_label = "Right" if repeats in [0, 1] else "Left"
    rep_num    = (repeats % 2) + 1

    # ── original state variables (unchanged) ──
    distances  = []
    elapsed_time = None
    start_time   = None
    last_detected_time = time.time()
    distance = None

    while True:
        if kinect.has_new_color_frame():
            image, results, frame = process_frame(kinect)

            distance = None  # reset before detection each frame

            if results.left_hand_landmarks and results.right_hand_landmarks:
                last_detected_time = time.time()

                draw_landmarks(image, results)

                hand_landmark1 = results.left_hand_landmarks.landmark[12]
                hand_landmark2 = results.right_hand_landmarks.landmark[12]

                right_hand = int(hand_landmark1.x * 640), int(hand_landmark1.y * 480)
                left_hand  = int(hand_landmark2.x * 640), int(hand_landmark2.y * 480)

                distance_pixel = calculate_distance_2d(right_hand, left_hand)
                distance       = (distance_pixel * PIXEL_TO_CM_RATIO) - ERROR

                elapsed_time, start_time = check_distance(distance, start_time)

                # original text (kept as-is)
                cv2.putText(image, f"Dist: {distance:.2f} cm",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(image, f'Pos Right Hand: {right_hand[0]}, {right_hand[1]}',
                            (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
                cv2.putText(image, f'Pos Left Hand: {left_hand[0]}, {left_hand[1]}',
                            (1000, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

                if elapsed_time >= POSE_HELD_DURATION:
                    distance = -distance
                    return f'{distance:.2f}'

            else:
                if time.time() - last_detected_time >= POSE_NO_HELD_DURATION:
                    start_time = None

            # ── v2 overlays (added on top) ──
            calib_state = _hands_calib_state(results, distance)
            draw_header(image, "Back Scratch", side_label, rep_num, 2)
            draw_calibration_legend(image, calib_state)

            cv2.imshow('Left Hand Tracking with Kinect and Holistic', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                finish_program()

repeats = 0
distances_right = []
distances_left  = []

age, height, weight, gender = register()
gender = "Feminine" if gender == "F" else "Male"

while repeats < 4:
    side_label = "Right" if repeats in [0, 1] else "Left"
    rep_num    = (repeats % 2) + 1

    final_distance = process_exercise(repeats)

    if final_distance is not None:
        real = real_distance()
        erro = np.abs(np.abs(float(real)) - np.abs(float(final_distance)))

        caminho_arquivo = "D:/CAPACITA/Propicie_visual/tabelas_utentes/back_scratch_utentes.xlsx"
        df = pd.read_excel(caminho_arquivo, engine="openpyxl")

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

        with open("D:/CAPACITA/Propicie_visual/logs_utentes/logs_back_scratch_utentes", "a") as arquivo:
            arquivo.write(f"{dt.datetime.now()}, {age}, {height}, {weight}, "
                          f"{gender}, {real}, {final_distance}, {side}\n")

        repeats += 1
        final_repetition_visualization(final_distance, real,
                                       "Back Scratch", side_label, rep_num)
    else:
        print("Exercise not performed correctly")
        finish_program()

best_left  = max(distances_left,  key=float)
best_right = max(distances_right, key=float)

# emit results for runner BEFORE opening final window
print(f"BS_RIGHT={best_right}")
print(f"BS_LEFT={best_left}")
sys.stdout.flush()

final_visualization(best_left, best_right)