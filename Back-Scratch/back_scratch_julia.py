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

# ─────────────────────────────────────────────
#  IDENTIDADE VISUAL CAPACITA — paleta BGR
# ─────────────────────────────────────────────
C_BG        = (247, 240, 234)   # #EAF0F7
C_PRIMARY   = (143,  46,  45)   # #2D2E8F  azul escuro
C_SECONDARY = (184,  78,  60)   # #3C4EB8  azul médio
C_WHITE     = (255, 255, 255)
C_DARK_TEXT = ( 45,  46, 141)   # texto em fundos claros
C_LIGHT_TXT = (200, 200, 220)
C_SUCCESS   = ( 80, 180,  80)
C_WARN      = ( 50, 140, 220)
C_ERROR     = ( 60,  60, 190)

# ─────────────────────────────────────────────
#  FONT CACHE
# ─────────────────────────────────────────────
_font_cache = {}

def get_font(size):
    """Returns TrueType font from cache or loads if necessary."""
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(fontFile, size)
        except IOError:
            # Fallback to default font if file doesn't exist
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

# ─────────────────────────────────────────────
#  HELPERS DE TEXTO UTF-8
# ─────────────────────────────────────────────
def put_text_utf8(img, text, pos, font_size, color_bgr, thickness=1):
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    
    # Convert OpenCV image to PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Get font from cache
    font = get_font(font_size)
    
    # Calculate adjusted position (PIL uses top-left corner)
    # Get font metrics to adjust baseline
    bbox = draw.textbbox((0, 0), text, font=font)
    text_height = bbox[3] - bbox[1]
    adjusted_y = pos[1] - text_height
    
    # Draw the text
    draw.text((pos[0], adjusted_y), text, font=font, fill=color_rgb)
    
    # Convert back to OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_text_size_utf8(text, font_size):
    font = get_font(font_size)
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def batch_put_text_utf8(img_bgr, text_items):
    """Batch render com uma única conversão BGR↔RGB."""
    if not text_items:
        return img_bgr
    
    # Single conversion: BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Batch all text operations in a single PIL draw session
    for text, org, font_size, color in text_items:
        font = get_font(font_size)
        rgb_color = (color[2], color[1], color[0])  # BGR → RGB
        draw.text(org, text, font=font, fill=rgb_color)
    
    # Single conversion: RGB → BGR
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img_bgr, result)
    return img_bgr


# ─────────────────────────────────────────────────────────────────
#  KINECT / MEDIAPIPE
# ─────────────────────────────────────────────────────────────────
kinect            = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic       = mp.solutions.holistic
holistic          = mp_holistic.Holistic()


# ─────────────────────────────────────────────────────────────────
#  PROCESS FRAME  (Phase 1 optimization — intacta)
# ─────────────────────────────────────────────────────────────────
def process_frame(kinect):
    frame = kinect.get_last_color_frame()
    frame = frame.reshape((1080, 1920, 4))
    # Direct BGRA → RGB conversion for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    results = holistic.process(rgb_frame)
    # Convert to BGR for OpenCV display
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), results, frame


# ─────────────────────────────────────────────────────────────────
#  OVERLAYS — IDENTIDADE VISUAL CAPACITA
# ─────────────────────────────────────────────────────────────────
def draw_header(img, exercise_label, side_label, rep_num, total_reps):
    h, w = img.shape[:2]
    header_h = 72
    cv2.rectangle(img, (0, 0), (w, header_h), C_PRIMARY, -1)
    cv2.line(img, (0, header_h), (w, header_h), C_SECONDARY, 3)

    text_items = []
    text_items.append((exercise_label.upper(), (24, 20), 26, C_WHITE))

    side_text = f"{_('Side')}: {side_label}"
    tw, _ = get_text_size_utf8(side_text, 26)
    text_items.append((side_text, (w // 2 - tw // 2, 20), 26, C_WHITE))

    rep_text = f"{_('Rep')} {rep_num}/{total_reps}"
    tw2, _ = get_text_size_utf8(rep_text, 26)
    text_items.append((rep_text, (w - tw2 - 24, 20), 26, C_WHITE))

    batch_put_text_utf8(img, text_items)
    return img


def draw_calibration_legend(img, calib_status):
    """
    Painel de calibração no canto inferior esquerdo.
    States: 'Not Detected' | 'Detected' | 'Ok'
    Estilo: CAPACITA
    """
    h, w = img.shape[:2]
    px, py  = 16, h - 230
    panel_w = 440
    panel_h = 215

    overlay = img.copy()
    cv2.rectangle(overlay, (px - 8, py - 8),
                  (px + panel_w, py + panel_h), C_BG, -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    cv2.rectangle(img, (px - 8, py - 8),
                  (px + panel_w, py + panel_h), C_PRIMARY, 2)

    text_items = []
    text_items.append((_("CALIBRATION STATUS"), (px, py + 4), 16, C_LIGHT_TXT))
    cv2.line(img, (px, py + 30), (px + panel_w - 8, py + 30), C_LIGHT_TXT, 1)

    states = [
        (_("Not Detected"), C_ERROR,   _("Position both hands in view")),
        (_("Detected"),     C_WARN,    _("Move hands closer together")),
        (_("Ok"),           C_SUCCESS, _("Hold position to measure")),
    ]

    for i, (label, col, hint) in enumerate(states):
        row_y  = py + 60 + i * 54
        active = (calib_status == label)

        sq_col = col if active else C_LIGHT_TXT
        cv2.rectangle(img, (px, row_y - 18), (px + 14, row_y + 8), sq_col, -1)
        if active:
            cv2.rectangle(img, (px - 1, row_y - 19), (px + 15, row_y + 9), C_PRIMARY, 1)

        txt_col  = C_DARK_TEXT if active else C_LIGHT_TXT
        font_sz  = 22 if active else 17
        text_items.append((label, (px + 24, row_y - 18), font_sz, txt_col))

        hint_col = col if active else C_LIGHT_TXT
        text_items.append((hint, (px + 24, row_y + 8), 14, hint_col))

        if active:
            text_items.append(("◀", (px + panel_w - 30, row_y - 12), 18, col))

    batch_put_text_utf8(img, text_items)
    return img


# ─────────────────────────────────────────────────────────────────
#  ECRÃS BASE — IDENTIDADE VISUAL CAPACITA
# ─────────────────────────────────────────────────────────────────
def _capacita_bg(h=600, w=960):
    img = np.ones((h, w, 3), dtype=np.uint8)
    img[:] = C_BG
    return img


def _draw_capacita_frame(img, title_text, subtitle=None):
    h, w = img.shape[:2]

    op_text = "IPBeja / Operador"
    tw, _ = get_text_size_utf8(op_text, 20)
    img = put_text_utf8(img, op_text, (w - tw - 30, 44), font_size=20, color_bgr=C_DARK_TEXT)

    title_x = 60
    title_y  = 30
    tw2, th2 = get_text_size_utf8(title_text, 44)
    img = put_text_utf8(img, title_text, (title_x + 20, title_y + th2), font_size=44, color_bgr=C_PRIMARY)

    cv2.line(img, (title_x, title_y + th2 // 2 + 4),
             (title_x + 16, title_y + th2 // 2 + 4), C_PRIMARY, 3)
    cv2.line(img, (title_x + 20 + tw2 + 12, title_y + th2 // 2 + 4),
             (w - 60, title_y + th2 // 2 + 4), C_PRIMARY, 3)

    content_top = title_y + th2 + 24

    box_x1, box_y1 = 60, content_top
    box_x2, box_y2 = w - 60, h - 60
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), C_PRIMARY, 2)

    if subtitle:
        sub_y = h - 50
        cv2.line(img, (box_x1, sub_y - 10), (box_x2, sub_y - 10), C_PRIMARY, 1)
        img = put_text_utf8(img, subtitle, (box_x1 + 20, sub_y), font_size=18, color_bgr=C_DARK_TEXT)

    return img, box_x1 + 20, box_y1 + 20, box_x2 - 20, box_y2 - 20


def _draw_result_block(img, x1, y1, x2, label, value):
    bar_h = 38
    val_h = 52
    cv2.rectangle(img, (x1, y1), (x2 - 20, y1 + bar_h), C_PRIMARY, -1)
    img = put_text_utf8(img, label, (x1 + 14, y1 + bar_h - 6), font_size=22, color_bgr=C_WHITE)
    cv2.rectangle(img, (x1, y1 + bar_h), (x2 - 20, y1 + bar_h + val_h), C_WHITE, -1)
    cv2.rectangle(img, (x1, y1 + bar_h), (x2 - 20, y1 + bar_h + val_h), C_SECONDARY, 1)
    img = put_text_utf8(img, value, (x1 + 14, y1 + bar_h + val_h - 10), font_size=28, color_bgr=C_DARK_TEXT)
    return img


def _base_screen(title_text, lines, prompt, subtitle=None):
    img = _capacita_bg()
    img, cx1, cy1, cx2, cy2 = _draw_capacita_frame(img, title_text, subtitle)

    for i, (text, color) in enumerate(lines):
        img = _draw_result_block(img, cx1, cy1 + 10 + i * 110, cx2, text, "")

    cv2.line(img, (60, 540), (900, 540), C_LIGHT_TXT, 1)
    pw, _ = get_text_size_utf8(prompt, 20)
    img = put_text_utf8(img, prompt, (480 - pw // 2, 565), font_size=20, color_bgr=C_DARK_TEXT)
    return img


def finish_program():
    cv2.destroyAllWindows()
    kinect.close()
    sys.exit(0)


def calculate_distance_2d(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def average_distance(distances):
    return sum(distances) / len(distances)


def final_repetition_visualization(distance, real_distance,
                                   exercise_label, side_label, rep_num):
    subtitle = f"{exercise_label}  |  {_('Side')}: {side_label}  |  {_('Rep')} {rep_num}"
    img = _capacita_bg()
    img, cx1, cy1, cx2, cy2 = _draw_capacita_frame(img, _("Repetition Completed"), subtitle=subtitle)

    img = _draw_result_block(img, cx1, cy1 + 10,  cx2,
                             _("Distance between both hands"), f"{distance} cm")
    img = _draw_result_block(img, cx1, cy1 + 130, cx2,
                             _("Real Distance"), f"{real_distance} cm")

    prompt = _('Press  "C"  to continue  |  "Q"  to quit')
    pw, _ = get_text_size_utf8(prompt, 20)
    img = put_text_utf8(img, prompt, (480 - pw // 2, 570), font_size=20, color_bgr=C_DARK_TEXT)

    cv2.imshow("Repetition Results", img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            finish_program()
        elif key == ord('c'):
            cv2.destroyWindow("Repetition Results"); break


def final_visualization(left, right):
    img = _capacita_bg()
    img, cx1, cy1, cx2, cy2 = _draw_capacita_frame(img, _("Exercise Completed"))

    img = _draw_result_block(img, cx1, cy1 + 10,  cx2,
                             _("Best result of the right side"), f"{right} cm")
    img = _draw_result_block(img, cx1, cy1 + 130, cx2,
                             _("Best result of the left side"),  f"{left} cm")

    prompt = _('Press  "Q"  to finish')
    pw, _ = get_text_size_utf8(prompt, 20)
    img = put_text_utf8(img, prompt, (480 - pw // 2, 570), font_size=20, color_bgr=C_DARK_TEXT)

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
    """Formulário de registo com identidade visual CAPACITA."""
    fields = [_("Age"), _("Height (cm)"), _("Weight (kg)"), _("Gender (M/F)")]
    values = ["", "", "", ""]
    active_field = -1

    field_x1, field_x2 = 280, 700
    field_ys = [220, 320, 420, 520]

    def mouse_callback(event, x, y, flags, param):
        nonlocal active_field
        if event == cv2.EVENT_LBUTTONDOWN:
            active_field = -1
            for i, fy in enumerate(field_ys):
                if field_x1 <= x <= field_x2 and fy - 30 <= y <= fy + 50:
                    active_field = i; break

    win_title = _("Registration")
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 840, 700)
    cv2.setMouseCallback(win_title, mouse_callback)

    while True:
        img = _capacita_bg(700, 840)
        img, cx1, cy1, cx2, cy2 = _draw_capacita_frame(img, _("Registration"))

        # Cabeçalho da caixa
        cv2.rectangle(img, (cx1, cy1), (cx2, cy1 + 46), C_PRIMARY, -1)
        tw, _ = get_text_size_utf8(_("Registration").upper(), 26)
        img = put_text_utf8(img, _("Registration").upper(),
                            ((cx1 + cx2) // 2 - tw // 2, cy1 + 38), font_size=26, color_bgr=C_WHITE)

        for i, fy in enumerate(field_ys):
            label_y = fy - 28
            field_y1, field_y2 = fy, fy + 50

            img = put_text_utf8(img, f"{fields[i]}:", (field_x1, label_y),
                                font_size=18, color_bgr=C_DARK_TEXT)

            border_col = C_PRIMARY if i == active_field else C_SECONDARY
            bw = 2 if i == active_field else 1
            cv2.rectangle(img, (field_x1, field_y1), (field_x2, field_y2), C_WHITE, -1)
            cv2.rectangle(img, (field_x1, field_y1), (field_x2, field_y2), border_col, bw)
            img = put_text_utf8(img, values[i], (field_x1 + 10, field_y2 - 10),
                                font_size=24, color_bgr=C_DARK_TEXT)

        img = put_text_utf8(img, _("Press Enter to finish"), (50, 650),
                            font_size=17, color_bgr=C_LIGHT_TXT)

        cv2.imshow(win_title, img)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:   finish_program()
        elif key in (13, 10): cv2.destroyAllWindows(); return values
        elif key == 9:  active_field = (active_field + 1) % len(fields)
        elif active_field != -1:
            if key == 8: values[active_field] = values[active_field][:-1]
            elif 32 <= key <= 126: values[active_field] += chr(key)


def real_distance():
    """Ecrã de entrada de distância real. Estilo CAPACITA."""
    distancia = ""
    win_title = _("Real Distance")
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 700, 420)

    while True:
        img = _capacita_bg(420, 700)
        img, cx1, cy1, cx2, cy2 = _draw_capacita_frame(img, _("Real Distance"))

        cv2.rectangle(img, (cx1, cy1), (cx2, cy1 + 42), C_PRIMARY, -1)
        label = _("Enter the measured distance (cm):")
        img = put_text_utf8(img, label, (cx1 + 14, cy1 + 34), font_size=20, color_bgr=C_WHITE)

        fi_y1, fi_y2 = cy1 + 60, cy1 + 120
        cv2.rectangle(img, (cx1, fi_y1), (cx2, fi_y2), C_WHITE, -1)
        cv2.rectangle(img, (cx1, fi_y1), (cx2, fi_y2), C_PRIMARY, 2)
        img = put_text_utf8(img, distancia, (cx1 + 16, fi_y2 - 8), font_size=36, color_bgr=C_DARK_TEXT)

        pw, _ = get_text_size_utf8(_("Press Enter to confirm"), 17)
        img = put_text_utf8(img, _("Press Enter to confirm"),
                            (350 - pw // 2, 380), font_size=17, color_bgr=C_LIGHT_TXT)

        cv2.imshow(win_title, img)
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
    if not (results.left_hand_landmarks and results.right_hand_landmarks):
        return _("Not Detected")
    if distance is None or distance >= DISTANCE_CHECK:
        return _("Detected")
    return _("Ok")


# ─────────────────────────────────────────────────────────────────
#  CICLO PRINCIPAL  (lógica de processamento intacta)
# ─────────────────────────────────────────────────────────────────
def process_exercise(repeats):
    side_label = _("Right") if repeats in [0, 1] else _("Left")
    rep_num    = (repeats % 2) + 1

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

                # PHASE 2 OPTIMIZATION: Batch text rendering for debug info
                debug_texts = [
                    (f"{_('Dist')}: {distance:.2f} cm", (50, 100), 24, C_WHITE),
                    (f'{_("Pos Right Hand")}: {right_hand[0]}, {right_hand[1]}', (1000, 100), 22, C_WHITE),
                    (f'{_("Pos Left Hand")}: {left_hand[0]}, {left_hand[1]}', (1000, 200), 22, C_WHITE),
                ]
                batch_put_text_utf8(image, debug_texts)

                if elapsed_time >= POSE_HELD_DURATION:
                    distance = -distance
                    return f'{distance:.2f}'

            else:
                if time.time() - last_detected_time >= POSE_NO_HELD_DURATION:
                    start_time = None

            calib_state = _hands_calib_state(results, distance)
            image = draw_header(image, _("Back Scratch"), side_label, rep_num, 2)
            image = draw_calibration_legend(image, calib_state)

            cv2.imshow('Left Hand Tracking with Kinect and Holistic', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                finish_program()


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
repeats = 0
distances_right = []
distances_left  = []

age, height, weight, gender = register()
gender = _("Feminine") if gender == "F" else _("Male")

while repeats < 4:
    side_label = _("Right") if repeats in [0, 1] else _("Left")
    rep_num    = (repeats % 2) + 1

    final_distance = process_exercise(repeats)

    if final_distance is not None:
        real = real_distance()
        erro = np.abs(np.abs(float(real)) - np.abs(float(final_distance)))

        caminho_arquivo = "./tabelas_utentes/back_scratch_utentes.xlsx"
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

        with open("./logs_utentes/logs_back_scratch_utentes", "a") as arquivo:
            arquivo.write(f"{dt.datetime.now()}, {age}, {height}, {weight}, "
                          f"{gender}, {real}, {final_distance}, {side}\n")

        repeats += 1
        final_repetition_visualization(final_distance, real,
                                       _("Back Scratch"), side_label, rep_num)
    else:
        print("Exercise not performed correctly")
        finish_program()

best_left  = max(distances_left,  key=float)
best_right = max(distances_right, key=float)

print(f"BS_RIGHT={best_right}")
print(f"BS_LEFT={best_left}")
sys.stdout.flush()

final_visualization(best_left, best_right)