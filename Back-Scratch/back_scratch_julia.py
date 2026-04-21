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
#  FONT CACHE (loads once per size)
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
#  HELPER FUNCTION: putText with UTF-8 support
# ─────────────────────────────────────────────
def put_text_utf8(img, text, pos, font_size, color_bgr, thickness=1):
    """
    Draws UTF-8 text on an OpenCV image using PIL.
    
    Args:
        img: OpenCV image (numpy array BGR)
        text: String to be drawn (supports UTF-8)
        pos: Tuple (x, y) - position of text
        font_size: Font size in pixels
        color_bgr: Color in BGR format (OpenCV)
        thickness: Ignored, kept for signature compatibility
    
    Returns:
        Modified image
    """
    # Convert BGR (OpenCV) to RGB (PIL)
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
    """
    Returns the text size in pixels.
    
    Args:
        text: String to be measured
        font_size: Font size
    
    Returns:
        Tuple (width, height)
    """
    font = get_font(font_size)
    # Use dummy image to calculate
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


# ─────────────────────────────────────────────────────────────────
#  OPTIMIZATION: Batch text rendering (PHASE 2)
# ─────────────────────────────────────────────────────────────────
def batch_put_text_utf8(img_bgr, text_items):
    """
    OPTIMIZED: Batch render multiple texts with a single BGR↔RGB conversion.
    
    Args:
        img_bgr: OpenCV image (BGR, uint8)
        text_items: List of tuples (text, org, font_size, color_bgr)
                   where org=(x,y), font_size=int, color_bgr=BGR tuple
    
    Returns:
        Modified image (operates in-place)
    """
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


# ─────────────────────────────────────────────────────────────────
#  OPTIMIZATION: Process frame with fewer conversions (PHASE 1)
# ─────────────────────────────────────────────────────────────────
def process_frame(kinect):
    """
    PHASE 1 OPTIMIZATION: Eliminated redundant color space conversions.
    
    Old flow: BGRA → BGR → RGB → BGR (3 conversions)
    New flow: BGRA → RGB → BGR (1 conversion)
    """
    frame = kinect.get_last_color_frame()
    frame = frame.reshape((1080, 1920, 4))
    # Direct BGRA → RGB conversion for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    results = holistic.process(rgb_frame)
    # Convert to BGR for OpenCV display
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), results, frame


# ─────────────────────────────────────────────────────────────────
#  OPTIMIZATION: draw_header with batch rendering
# ─────────────────────────────────────────────────────────────────
def draw_header(img, exercise_label, side_label, rep_num, total_reps):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 70), C_PANEL, -1)
    cv2.line(img, (0, 70), (w, 70), C_ACCENT, 2)
    
    # PHASE 2 OPTIMIZATION: Batch text rendering
    text_items = []
    
    # Exercise label
    text_items.append((exercise_label, (20, 14), 32, C_ACCENT))
    
    # Side label
    side_text = f"{_('Side')}: {side_label}"
    tw, fm = get_text_size_utf8(side_text, font_size=28)
    text_items.append((side_text, (w // 2 - tw // 2, 18), 28, C_YELLOW))
    
    # Rep counter
    rep_text = f"{_('Rep')} {rep_num}/{total_reps}"
    tw2, fm2 = get_text_size_utf8(rep_text, font_size=28)
    text_items.append((rep_text, (w - tw2 - 20, 18), 28, C_WHITE))
    
    # Batch render all text at once
    batch_put_text_utf8(img, text_items)
    return img


# ─────────────────────────────────────────────────────────────────
#  OPTIMIZATION: draw_calibration_legend with batch rendering
# ─────────────────────────────────────────────────────────────────
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

    # PHASE 2 OPTIMIZATION: Prepare all text for batch rendering
    text_items = []
    text_items.append((_("CALIBRATION STATUS"), (px, py + 2), 18, C_GREY))
    
    cv2.line(img, (px, py + 28), (px + panel_w - 10, py + 28), C_GREY, 1)

    states = [
        (_("Not Detected"), C_ERROR,   _("Position both hands in view")),
        (_("Detected"),     C_WARN,    _("Move hands closer together")),
        (_("Ok"),           C_SUCCESS, _("Hold position to measure")),
    ]

    for i, (label, col, hint) in enumerate(states):
        row_y  = py + 58 + i * 52
        active = (calib_status == label)

        sq_col = col if active else (50, 50, 70)
        cv2.rectangle(img, (px, row_y - 18), (px + 16, row_y + 6), sq_col, -1)
        if active:
            cv2.rectangle(img, (px, row_y - 18), (px + 16, row_y + 6), C_WHITE, 1)

        txt_col   = C_WHITE if active else C_GREY
        font_sz   = 22 if active else 18
        text_items.append((label, (px + 26, row_y - 18), font_sz, txt_col))

        hint_col = col if active else (65, 65, 85)
        text_items.append((hint, (px + 26, row_y + 6), 15, hint_col))

        if active:
            text_items.append(("<<", (px + panel_w - 40, row_y - 14), 20, col))
    
    # Batch render all text at once
    batch_put_text_utf8(img, text_items)
    
    return img

def _gradient_bg(h=500, w=900):
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    return img


def _base_screen(title_text, lines, prompt):
    img = _gradient_bg()
    cv2.rectangle(img, (0, 0), (900, 6), C_ACCENT, -1)
    (tw, fm) = get_text_size_utf8(title_text, font_size=36)
    img = put_text_utf8(img, title_text, (900 // 2 - tw // 2, 80),
                        font_size=36, color_bgr=(0, 0, 0))
    cv2.line(img, (60, 100), (840, 100), C_ACCENT, 1)
    for i, (text, color) in enumerate(lines):
        img = put_text_utf8(img, text, (60, 155 + i * 55),
                            font_size=24, color_bgr=color)
    cv2.line(img, (60, 420), (840, 420), C_GREY, 1)
    (pw, dm) = get_text_size_utf8(prompt, font_size=19)
    img = put_text_utf8(img, prompt, (900 // 2 - pw // 2, 465),
                        font_size=19, color_bgr=(0, 0, 0))
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
    lines = [
        (f"{_('Distance between both hands')}: {distance} cm", C_SUCCESS),
        (f"{_('Real Distance')}: {real_distance} cm", C_ACCENT),
        (f"{exercise_label}  |  {_('Side')}: {side_label}  |  {_('Rep')} {rep_num}", C_GREY),
    ]
    img = _base_screen(_("Repetition Completed"), lines,
                       _(f'Press  "C"  to continue  |  "Q"  to quit'))
    cv2.imshow("Repetition Results", img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): finish_program()
        elif key == ord('c'): cv2.destroyWindow("Repetition Results"); break

def final_visualization(left, right):
    lines = [
        (f"{_('Best result of the right side')}: {right} cm", C_SUCCESS),
        (f"{_('Best result of the left side')}: {left}  cm", C_SUCCESS),
    ]
    img = _base_screen(_("Exercise Completed"), lines, _(f'Press  "Q"  to finish'))
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
    fields = [_("Age"), _("Height (cm)"), _("Weight (kg)"), _("Gender (M/F)")]
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

    cv2.namedWindow(_("Registration"))
    cv2.setMouseCallback(_("Registration"), mouse_callback)
    while True:
        img = 255 * np.ones((400, 600, 3), dtype=np.uint8)
        for i, (x1, y1, x2, y2) in enumerate(positions):
            cv2.rectangle(img, (x1, y1), (x2, y2), (230, 230, 230), -1)
            border_color = (0, 255, 0) if i == active_field else (0, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2)
            img = put_text_utf8(img, f"{fields[i]}:", (x1 + 10, y1 - 10),
                                font_size=15, color_bgr=(0, 0, 0))
            img = put_text_utf8(img, values[i], (x1 + 10, y2 - 20),
                                font_size=20, color_bgr=(0, 0, 0))
        img = put_text_utf8(img, _("Press Enter to finish"), (50, 380),
                            font_size=15, color_bgr=(100, 100, 100))
        cv2.imshow(_("Registration"), img)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:   finish_program()
        elif key in (13, 10): cv2.destroyAllWindows(); return values
        elif key == 9:  active_field = (active_field + 1) % len(fields)
        elif active_field != -1:
            if key == 8: values[active_field] = values[active_field][:-1]
            elif 32 <= key <= 126: values[active_field] += chr(key)

def real_distance():
    distancia = ""
    cv2.namedWindow(_("Real Distance"))
    while True:
        img = np.ones((200, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 60), (550, 120), (230, 230, 230), -1)
        cv2.rectangle(img, (50, 60), (550, 120), (0, 0, 0), 2)
        img = put_text_utf8(img, _("Enter the measured distance (cm):"), (50, 40),
                            font_size=20, color_bgr=(0, 0, 0))
        img = put_text_utf8(img, distancia, (60, 105),
                            font_size=30, color_bgr=(0, 0, 255))
        img = put_text_utf8(img, _("Press Enter to confirm"), (50, 170),
                            font_size=15, color_bgr=(100, 100, 100))
        cv2.imshow(_("Real Distance"), img)
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
        return _("Not Detected")
    if distance is None or distance >= DISTANCE_CHECK:
        return _("Detected")
    return _("Ok")


# ─────────────────────────────────────────────────────────────────
#  OPTIMIZATION: process_exercise with batch text rendering
# ─────────────────────────────────────────────────────────────────
def process_exercise(repeats):
    side_label = _("Right") if repeats in [0, 1] else _("Left")
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

                # PHASE 2 OPTIMIZATION: Batch text rendering for debug info
                debug_texts = [
                    (f"{_('Dist')}: {distance:.2f} cm", (50, 30), 26, (0, 0, 0)),
                    (f'{_("Pos Right Hand")}: {right_hand[0]}, {right_hand[1]}', (1000, 100), 24, (0, 235, 0)),
                    (f'{_("Pos Left Hand")}: {left_hand[0]}, {left_hand[1]}', (1000, 200), 24, (0, 235, 0)),
                ]
                batch_put_text_utf8(image, debug_texts)

                if elapsed_time >= POSE_HELD_DURATION:
                    distance = -distance
                    return f'{distance:.2f}'

            else:
                if time.time() - last_detected_time >= POSE_NO_HELD_DURATION:
                    start_time = None

            # ── v2 overlays (added on top) ──
            calib_state = _hands_calib_state(results, distance)
            image = draw_header(image, _("Back Scratch"), side_label, rep_num, 2)
            image = draw_calibration_legend(image, calib_state)

            cv2.imshow('Left Hand Tracking with Kinect and Holistic', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                finish_program()


# ═════════════════════════════════════════════════════════════════
#  MAIN PROGRAM
# ═════════════════════════════════════════════════════════════════

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
                                       _("Back Scratch"), side_label, rep_num)
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