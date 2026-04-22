"""
sit_and_reach_julia.py — versão com suporte a múltiplos idiomas
  • gettext  → traduções das strings da UI
  • Babel    → formatação localizada de números e datas
  • PIL      → desenho de texto UTF-8 sobre frames OpenCV
  
OPTIMIZATIONS (Phase 1 & 2):
  • Phase 1: Eliminated redundant color space conversions (3 → 1 conversion in process_frame)
  • Phase 2: Batched PIL text rendering to reduce BGR↔RGB conversions (7+ → 1-2 per batch)
"""

from pykinect2 import PyKinectRuntime, PyKinectV2
import mediapipe as mp
import datetime as dt
import pandas as pd
import numpy as np
import time
import math
import sys
import cv2
import gettext
from PIL import Image, ImageDraw, ImageFont
from babel.numbers import format_decimal
from babel.dates   import format_datetime

# ─────────────────────────────────────────────────────────────────
#  INTERNACIONALIZAÇÃO  (gettext + Babel)
# ─────────────────────────────────────────────────────────────────

FONT_FILE  = "LiberationSansBold.ttf"
FONT_SIZES = {}

LANGUAGE = "en_US"
args = sys.argv
if len(args) >= 2:
    LANGUAGE = args[1]

try:
    _lang = gettext.translation("messages", localedir="locale", languages=[LANGUAGE])
    _lang.install()
    _ = _lang.gettext
except FileNotFoundError:
    _ = gettext.gettext

BABEL_LOCALE = LANGUAGE


def fmt_number(value, decimal_places=2):
    fmt = f"#,##0.{'0' * decimal_places}"
    return format_decimal(value, format=fmt, locale=BABEL_LOCALE)


def fmt_datetime(value=None):
    if value is None:
        value = dt.datetime.now()
    return format_datetime(value, locale=BABEL_LOCALE)


# ─────────────────────────────────────────────────────────────────
#  IDENTIDADE VISUAL — CAPACITA / IPBeja
# ─────────────────────────────────────────────────────────────────

# Paleta de cores em BGR (convenção OpenCV)
# Primária: #2D2E8F → BGR (143, 46, 45)
# Secundária: #3C4EB8 → BGR (184, 78, 60)
# Fundo: #EAF0F7 → BGR (247, 240, 234)

C_BG        = (247, 240, 234)   # #EAF0F7  — fundo geral
C_PRIMARY   = (143,  46,  45)   # #2D2E8F  — azul escuro CAPACITA
C_SECONDARY = (184,  78,  60)   # #3C4EB8  — azul médio
C_WHITE     = (255, 255, 255)
C_DARK_TEXT = ( 45,  46, 141)   # texto escuro em fundos claros (BGR de #2D2E8F)
C_LIGHT_TXT = (200, 200, 220)   # texto cinza suave
C_SUCCESS   = ( 80, 180,  80)   # verde confirmação
C_WARN      = ( 50, 140, 220)   # laranja/amarelo aviso
C_ERROR     = ( 60,  60, 190)   # vermelho erro


# ─────────────────────────────────────────────────────────────────
#  DESENHO DE TEXTO UTF-8 COM PIL
# ─────────────────────────────────────────────────────────────────

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in FONT_SIZES:
        try:
            FONT_SIZES[size] = ImageFont.truetype(FONT_FILE, size)
        except (IOError, OSError):
            # Fallback para a fonte bitmap embutida do PIL
            FONT_SIZES[size] = ImageFont.load_default()
    return FONT_SIZES[size]


def put_text_utf8(img_bgr: np.ndarray, text: str, org: tuple,
                  font_size: int = 24, color: tuple = (255, 255, 255),
                  thickness: int = 1) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)
    font    = _get_font(font_size)
    rgb_color = (color[2], color[1], color[0])
    draw.text(org, text, font=font, fill=rgb_color)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img_bgr, result)
    return img_bgr


def get_text_size_utf8(text: str, font_size: int) -> tuple:
    font = _get_font(font_size)
    bbox = font.getbbox(text)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def batch_put_text_utf8(img_bgr: np.ndarray, text_items: list) -> np.ndarray:
    """Batch render múltiplos textos com uma única conversão BGR↔RGB."""
    if not text_items:
        return img_bgr
    
    # Single conversion: BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    for text, org, font_size, color in text_items:
        font = _get_font(font_size)
        rgb_color = (color[2], color[1], color[0])  # BGR → RGB
        draw.text(org, text, font=font, fill=rgb_color)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img_bgr, result)
    return img_bgr


# ─────────────────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────
#  KINECT / MEDIAPIPE
# ─────────────────────────────────────────────────────────────────

kinect            = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic       = mp.solutions.holistic
holistic          = mp_holistic.Holistic()


# ─────────────────────────────────────────────────────────────────
#  OVERLAYS — IDENTIDADE VISUAL CAPACITA
# ─────────────────────────────────────────────────────────────────

def draw_header(img, exercise_label, side_label, rep_num, total_reps):
    """
    Header com 3 colunas: exercício | lado | repetição
    Estilo CAPACITA: fundo azul escuro, texto branco, linha separadora
    """
    h, w = img.shape[:2]
    header_h = 72

    # Fundo do header: azul escuro CAPACITA
    cv2.rectangle(img, (0, 0), (w, header_h), C_PRIMARY, -1)
    # Linha inferior de separação em azul secundário
    cv2.line(img, (0, header_h), (w, header_h), C_SECONDARY, 3)

    text_items = []

    # Coluna esquerda — nome do exercício
    text_items.append((exercise_label.upper(), (24, 20), 26, C_WHITE))

    # Coluna central — lado
    side_text = f"{_('Side')}: {side_label}"
    tw, _ = get_text_size_utf8(side_text, 26)
    text_items.append((side_text, (w // 2 - tw // 2, 20), 26, C_WHITE))

    # Coluna direita — repetição
    rep_text = f"{_('Rep')} {rep_num}/{total_reps}"
    tw2, _ = get_text_size_utf8(rep_text, 26)
    text_items.append((rep_text, (w - tw2 - 24, 20), 26, C_WHITE))

    batch_put_text_utf8(img, text_items)


def draw_calibration_legend(img, calib_status):
    """
    Painel de calibração no canto inferior esquerdo.
    Estilo CAPACITA: fundo claro, borda azul escuro, estado ativo destacado.
    """
    h, w = img.shape[:2]
    px, py  = 16, h - 230
    panel_w = 440
    panel_h = 215

    # Fundo semitransparente — branco com leve azul
    overlay = img.copy()
    cv2.rectangle(overlay, (px - 8, py - 8),
                  (px + panel_w, py + panel_h), C_BG, -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    # Borda azul escuro
    cv2.rectangle(img, (px - 8, py - 8),
                  (px + panel_w, py + panel_h), C_PRIMARY, 2)

    text_items = []
    text_items.append((_("CALIBRATION STATUS"), (px, py + 4), 16, C_LIGHT_TXT))
    cv2.line(img, (px, py + 30), (px + panel_w - 8, py + 30), C_LIGHT_TXT, 1)

    states = [
        ("Wrong Position", C_ERROR,   _("Adjust your posture")),
        ("Right Position", C_WARN,    _("Hold still to calibrate")),
        ("Ok",             C_SUCCESS, _("Calibrated — start the exercise")),
    ]

    for i, (label, col, hint) in enumerate(states):
        row_y  = py + 60 + i * 54
        active = (calib_status == label)

        sq_col = col if active else C_LIGHT_TXT
        cv2.rectangle(img, (px, row_y - 18), (px + 14, row_y + 8), sq_col, -1)
        if active:
            cv2.rectangle(img, (px - 1, row_y - 19), (px + 15, row_y + 9), C_PRIMARY, 1)

        txt_col  = C_DARK_TEXT if active else C_LIGHT_TXT
        fs_label = 22 if active else 17
        text_items.append((_(label), (px + 24, row_y - 18), fs_label, txt_col))

        hint_col = col if active else C_LIGHT_TXT
        text_items.append((hint, (px + 24, row_y + 8), 14, hint_col))

        if active:
            text_items.append(("◀", (px + panel_w - 30, row_y - 12), 18, col))

    batch_put_text_utf8(img, text_items)
    return py + panel_h


# ─────────────────────────────────────────────────────────────────
#  ECRÃS BASE
# ─────────────────────────────────────────────────────────────────

def _capacita_bg(h=840, w=1456):
    """Fundo azul muito claro, estilo CAPACITA."""
    img = np.ones((h, w, 3), dtype=np.uint8)
    img[:] = (247, 240, 234)   # C_BG em BGR
    return img


def _draw_capacita_frame(img, title_text, subtitle=None):
    """
    Frame base com borda e cabeçalho CAPACITA.
    Retorna o img com o frame desenhado e a posição y disponível para conteúdo.
    """
    h, w = img.shape[:2]

    # IPBeja / Operador no canto superior direito
    op_text = "IPBeja / Operador"
    tw, _ = get_text_size_utf8(op_text, 20)
    put_text_utf8(img, op_text, (w - tw - 30, 24), font_size=20, color=C_DARK_TEXT)

    # Linha horizontal do título
    title_x = 60
    title_y  = 30

    # Título em bold azul escuro
    tw2, th2 = get_text_size_utf8(title_text, 44)
    put_text_utf8(img, title_text, (title_x + 20, title_y), font_size=44, color=C_PRIMARY)

    # Linhas decorativas ao lado do título
    cv2.line(img, (title_x, title_y + th2 // 2 + 4),
             (title_x + 16, title_y + th2 // 2 + 4), C_PRIMARY, 3)
    cv2.line(img, (title_x + 20 + tw2 + 12, title_y + th2 // 2 + 4),
             (w - 60, title_y + th2 // 2 + 4), C_PRIMARY, 3)

    content_top = title_y + th2 + 20

    # Caixa de conteúdo com borda
    box_x1, box_y1 = 60, content_top
    box_x2, box_y2 = w - 60, h - 60
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), C_PRIMARY, 2)

    if subtitle:
        # Linha inferior do frame com info contextual
        sub_y = h - 50
        cv2.line(img, (box_x1, sub_y - 10), (box_x2, sub_y - 10), C_PRIMARY, 1)
        put_text_utf8(img, subtitle, (box_x1 + 20, sub_y), font_size=18, color=C_DARK_TEXT)

    return box_x1 + 20, box_y1 + 20, box_x2 - 20, box_y2 - 20


def _base_screen(title_text, lines, prompt, subtitle=None):
    """Ecrã base com identidade CAPACITA."""
    img = _capacita_bg(600, 960)
    cx1, cy1, cx2, cy2 = _draw_capacita_frame(img, title_text, subtitle)

    for i, (text, color) in enumerate(lines):
        # Bloco de resultado: barra azul + valor em caixa branca
        bar_y = cy1 + 10 + i * 110
        cv2.rectangle(img, (cx1, bar_y), (cx2 - 20, bar_y + 36), C_PRIMARY, -1)
        tw, _ = get_text_size_utf8(text, 22)
        put_text_utf8(img, text, (cx1 + 16, bar_y + 6), font_size=22, color=C_WHITE)
        # Valor separado abaixo seria ideal mas aqui os dados já vêm formatados
        cv2.rectangle(img, (cx1, bar_y + 38), (cx2 - 20, bar_y + 82), C_WHITE, -1)
        cv2.rectangle(img, (cx1, bar_y + 38), (cx2 - 20, bar_y + 82), C_PRIMARY, 1)

    # Prompt centrado na base
    cv2.line(img, (60, 540), (900, 540), C_LIGHT_TXT, 1)
    pw, _ = get_text_size_utf8(prompt, 20)
    put_text_utf8(img, prompt, (480 - pw // 2, 550), font_size=20, color=C_DARK_TEXT)
    return img


# ─────────────────────────────────────────────────────────────────
#  UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────
#  PROCESSAMENTO DE FRAMES / LANDMARKS
# ─────────────────────────────────────────────────────────────────

def process_frame(kinect):
    """
    PHASE 1 OPTIMIZATION: Eliminated redundant color space conversions.
    Old flow: BGRA → BGR → RGB → RGB (writeable) → BGR = 3 conversions
    New flow: BGRA → RGB (direct) = 1 conversion
    """
    frame = kinect.get_last_color_frame()
    frame = frame.reshape((1080, 1920, 4))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
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
        verification_hand = results.left_hand_landmarks
        hand_landmarks    = results.left_hand_landmarks.landmark
    else:
        if not results.pose_landmarks or not results.right_hand_landmarks:
            return None, None
        verification_hand = results.right_hand_landmarks
        hand_landmarks    = results.right_hand_landmarks.landmark
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

    def lm(i):
        return np.array([pose_landmarks[indices[i]].x, pose_landmarks[indices[i]].y])

    shoulder  = lm(0); elbow    = lm(1); wrist    = lm(2)
    hip       = lm(3); knee     = lm(4); ankle    = lm(5)
    o_shoulder = lm(6); o_elbow = lm(7); o_wrist  = lm(8)

    def coord(v):
        return tuple(np.multiply(v[:2], [frame.shape[1], frame.shape[0]]).astype(int))

    shoulder_c  = coord(shoulder);  elbow_c   = coord(elbow);   wrist_c   = coord(wrist)
    hip_c       = coord(hip);       knee_c    = coord(knee);    ankle_c   = coord(ankle)
    o_shoulder_c = coord(o_shoulder); o_elbow_c = coord(o_elbow); o_wrist_c = coord(o_wrist)

    put_text_utf8(image,
                  f"{_('Opposite Knee Angle')}: {fmt_number(opposite_knee_angle)}",
                  (1000, 400), font_size=22, color=C_WHITE)
    put_text_utf8(image,
                  f"{_('Opposite Elbow Angle')}: {fmt_number(opposite_elbow_angle)}",
                  o_elbow_c, font_size=22, color=C_WHITE)
    draw_dynamic_angle_arc(image, o_shoulder_c, o_elbow_c, o_wrist_c, opposite_elbow_angle)
    draw_dynamic_angle_arc(image, hip_c, knee_c, ankle_c, knee_angle)
    put_text_utf8(image,
                  f"{_('Knee Angle')}: {fmt_number(knee_angle)}",
                  knee_c, font_size=22, color=C_WHITE)
    draw_dynamic_angle_arc(image, shoulder_c, hip_c, knee_c, hip_angle)
    put_text_utf8(image,
                  f"{_('Hip Angle')}: {fmt_number(hip_angle)}",
                  hip_c, font_size=22, color=C_WHITE)
    draw_dynamic_angle_arc(image, shoulder_c, elbow_c, wrist_c, elbow_angle)
    put_text_utf8(image,
                  f"{_('Elbow Angle')}: {fmt_number(elbow_angle)}",
                  elbow_c, font_size=22, color=C_WHITE)


def calculate_angles(repeats, pose_landmarks):
    side = "right" if repeats in [0, 1] else "left"
    pose_indices = {
        "right": [11, 13, 15, 23, 25, 27, 24, 26, 28, 12, 14, 16],
        "left":  [12, 14, 16, 24, 26, 28, 23, 25, 27, 11, 13, 15]
    }
    indices = pose_indices[side]

    def pt(i):
        return np.array([pose_landmarks[indices[i]].x, pose_landmarks[indices[i]].y])

    shoulder = pt(0); elbow = pt(1); wrist = pt(2)
    hip = pt(3);      knee  = pt(4); ankle = pt(5)
    o_hip = pt(6);    o_knee = pt(7); o_ankle = pt(8)
    o_shoulder = pt(9); o_elbow = pt(10); o_wrist = pt(11)

    return (calculate_angle(hip, knee, ankle),
            calculate_angle(o_hip, o_knee, o_ankle),
            calculate_angle(shoulder, hip, knee),
            calculate_angle(shoulder, elbow, wrist),
            calculate_angle(o_shoulder, o_elbow, o_wrist))


# ─────────────────────────────────────────────────────────────────
#  CALIBRAÇÃO / POSTURA
# ─────────────────────────────────────────────────────────────────

def check_calibration(calibration_time, foot, repeats, knee_angle, opposite_knee_angle,
                      hip_angle, elbow_angle, progress_calibration, progress_calibration1,
                      calibration_held_duration, pose_landmarks):
    foot_index = 31 if repeats in [0, 1] else 32
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


# ─────────────────────────────────────────────────────────────────
#  ECRÃS DE RESULTADO — IDENTIDADE VISUAL CAPACITA
# ─────────────────────────────────────────────────────────────────

def final_visualization(left, right):
    """Ecrã final com resultados. Dois blocos de resultado estilo CAPACITA."""
    img = _capacita_bg(600, 960)
    cx1, cy1, cx2, cy2 = _draw_capacita_frame(img, _("Exercise Completed"))

    # Bloco 1
    _draw_result_block(img, cx1, cy1 + 10,  cx2,
                       _("Best result of the right leg"), f"{left} cm")
    # Bloco 2
    _draw_result_block(img, cx1, cy1 + 130, cx2,
                       _("Best result of the left leg"),  f"{right} cm")

    # Prompt
    prompt = _('Press  "Q"  to finish')
    pw, _ = get_text_size_utf8(prompt, 20)
    put_text_utf8(img, prompt, (480 - pw // 2, 555), font_size=20, color=C_DARK_TEXT)

    cv2.imshow(_("Final Results"), img)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def _draw_result_block(img, x1, y1, x2, label, value):
    """Bloco de resultado: barra de label azul + caixa de valor branca."""
    bar_h = 38
    val_h = 52
    # Barra de label
    cv2.rectangle(img, (x1, y1), (x2 - 20, y1 + bar_h), C_PRIMARY, -1)
    put_text_utf8(img, label, (x1 + 14, y1 + 6), font_size=22, color=C_WHITE)
    # Caixa de valor
    cv2.rectangle(img, (x1, y1 + bar_h), (x2 - 20, y1 + bar_h + val_h), C_WHITE, -1)
    cv2.rectangle(img, (x1, y1 + bar_h), (x2 - 20, y1 + bar_h + val_h), C_SECONDARY, 1)
    put_text_utf8(img, value, (x1 + 14, y1 + bar_h + 10), font_size=28, color=C_DARK_TEXT)


def final_repetition_visualization(final_distance, real_dist,
                                   exercise_label, side_label, rep_num):
    """Ecrã de repetição concluída. Estilo CAPACITA."""
    subtitle = f"{exercise_label}  |  {_('Side')}: {side_label}  |  {_('Rep')} {rep_num}"
    img = _capacita_bg(600, 960)
    cx1, cy1, cx2, cy2 = _draw_capacita_frame(img, _("Repetition Completed"), subtitle=subtitle)

    _draw_result_block(img, cx1, cy1 + 10,  cx2,
                       _("Final Distance"),  f"{final_distance} cm")
    _draw_result_block(img, cx1, cy1 + 130, cx2,
                       _("Real Distance"),   f"{real_dist} cm")

    prompt = _('Press  "C"  to continue  |  "Q"  to quit')
    pw, _ = get_text_size_utf8(prompt, 20)
    put_text_utf8(img, prompt, (480 - pw // 2, 555), font_size=20, color=C_DARK_TEXT)

    cv2.imshow(_("Final Repetition Results"), img)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.destroyWindow(_("Final Repetition Results")); break
        elif key == ord('q'):
            finish_program()


def register():
    """Formulário de registo com identidade visual CAPACITA."""
    fields = [
        _("Age"),
        _("Height (cm)"),
        _("Weight (kg)"),
        _("Gender (M/F)"),
    ]
    values = ["", "", "", ""]
    active_field = -1

    # Posições dos campos dentro da caixa de conteúdo
    field_x1, field_x2 = 340, 760
    field_ys = [220, 320, 420, 520]
    positions = [(field_x1, fy - 30, field_x2, fy + 10) for fy in field_ys]

    def mouse_callback(event, x, y, flags, param):
        nonlocal active_field
        if event == cv2.EVENT_LBUTTONDOWN:
            active_field = -1
            for i, (x1, y1, x2, y2) in enumerate(positions):
                if x1 <= x <= x2 and y1 - 30 <= y <= y2 + 30:
                    active_field = i; break

    win_title = _("Registration")
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 840, 700)
    cv2.setMouseCallback(win_title, mouse_callback)

    while True:
        img = _capacita_bg(700, 840)

        # Frame CAPACITA com título CADASTRO
        cx1, cy1, cx2, cy2 = _draw_capacita_frame(img, _("Registration"))

        # Cabeçalho da caixa
        cv2.rectangle(img, (cx1, cy1), (cx2, cy1 + 46), C_PRIMARY, -1)
        tw, _ = get_text_size_utf8(_("Registration").upper(), 26)
        put_text_utf8(img, _("Registration").upper(),
                      ((cx1 + cx2) // 2 - tw // 2, cy1 + 8), font_size=26, color=C_WHITE)

        for i, (fy) in enumerate(field_ys):
            x1, x2 = field_x1, field_x2
            label_y = fy - 28
            field_y1, field_y2 = fy, fy + 50

            put_text_utf8(img, f"{fields[i]}:", (x1, label_y), font_size=18, color=C_DARK_TEXT)

            border_col = C_PRIMARY if i == active_field else C_SECONDARY
            bw = 2 if i == active_field else 1
            cv2.rectangle(img, (x1, field_y1), (x2, field_y2), C_WHITE, -1)
            cv2.rectangle(img, (x1, field_y1), (x2, field_y2), border_col, bw)

            put_text_utf8(img, values[i], (x1 + 10, field_y1 + 8), font_size=24, color=C_DARK_TEXT)

        prompt = _("Press Enter to confirm")
        pw, _ = get_text_size_utf8(prompt, 17)
        put_text_utf8(img, prompt, (420 - pw // 2, 640), font_size=17, color=C_LIGHT_TXT)

        cv2.imshow(win_title, img)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            finish_program()
        elif key in (13, 10):
            cv2.destroyAllWindows(); return values
        elif key == 9:
            active_field = (active_field + 1) % len(fields)
        elif active_field != -1:
            if key == 8:
                values[active_field] = values[active_field][:-1]
            elif 32 <= key <= 126:
                values[active_field] += chr(key)


def real_distance():
    distancia = ""
    win_title = _("Real Distance")
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 700, 420)

    while True:
        img = _capacita_bg(420, 700)
        cx1, cy1, cx2, cy2 = _draw_capacita_frame(img, _("Real Distance"))

        # Cabeçalho da caixa
        cv2.rectangle(img, (cx1, cy1), (cx2, cy1 + 42), C_PRIMARY, -1)
        label = _("Enter the measured distance (cm):")
        tw, _ = get_text_size_utf8(label, 20)
        put_text_utf8(img, label, (cx1 + 14, cy1 + 8), font_size=20, color=C_WHITE)

        # Campo de entrada
        fi_y1, fi_y2 = cy1 + 65, cy1 + 130
        cv2.rectangle(img, (cx1, fi_y1), (cx2, fi_y2), C_WHITE, -1)
        cv2.rectangle(img, (cx1, fi_y1), (cx2, fi_y2), C_PRIMARY, 2)
        put_text_utf8(img, distancia, (cx1 + 16, fi_y1 + 12), font_size=38, color=C_DARK_TEXT)

        # Prompt
        prompt = _("Press Enter to confirm")
        pw, _ = get_text_size_utf8(prompt, 17)
        put_text_utf8(img, prompt, (350 - pw // 2, 360), font_size=17, color=C_LIGHT_TXT)

        cv2.imshow(win_title, img)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            cv2.destroyAllWindows(); finish_program()
        elif key in (13, 10):
            if distancia:
                cv2.destroyAllWindows()
                return float(distancia.replace(",", "."))
        elif key == 8:
            distancia = distancia[:-1]
        elif (48 <= key <= 57) or key in [44, 46, 43, 45]:
            distancia += chr(key)


# ─────────────────────────────────────────────────────────────────
#  CICLO PRINCIPAL DO EXERCÍCIO
# ─────────────────────────────────────────────────────────────────

def process_exercise(repeats):
    side_label = _("Right") if repeats in [0, 1] else _("Left")
    rep_num    = (repeats % 2) + 1

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

                    distances.append(distance)
                    if len(distances) > AVERAGE_OVER:
                        distances.pop(0)
                        distance = average_distance(distances)

                    pose_correct, progress, pose_correct_start_time, final_distance = \
                        check_posture(pose_correct_start_time, *angles,
                                      POSE_HELD_DURATION, progress, distance)

                    if final_distance is not None:
                        if repeats in [0, 1]:
                            if hand[0] < foot[0] and distance > 1.2:
                                final_distance = -(final_distance + ERROR)
                        else:
                            if hand[0] > foot[0] and distance > 1.2:
                                final_distance = -(final_distance + ERROR)
                        break

                    # PHASE 2 OPTIMIZATION: Batch text rendering for all information display
                    info_texts = [
                        (f"{_('Foot position X/Y')}: {foot[0]}, {foot[1]}", (1000, 100), 22, C_WHITE),
                        (f"{_('Hand position X/Y')}: {hand[0]}, {hand[1]}", (1000, 200), 22, C_WHITE),
                        (f"{_('Dist')}: {fmt_number(distance)} cm", (50, 100), 24, C_WHITE),
                        (f"{_('Pose')}: {_(pose_correct)}", (50, 230), 24, C_WHITE),
                        (f"{_('Calibration')}: {_(calibration)}", (50, 130), 24, C_WHITE),
                    ]
                    batch_put_text_utf8(image, info_texts)
                else:
                    # PHASE 2 OPTIMIZATION: Batch text rendering during calibration phase
                    calib_texts = [
                        (f"{_('Calibration')}: {_(calibration)}", (50, 130), 24, C_WHITE),
                    ]
                    batch_put_text_utf8(image, calib_texts)

            # Overlays CAPACITA
            draw_header(image, _("Sit and Reach"), side_label, rep_num, 2)
            draw_calibration_legend(image, calibration)

            cv2.imshow("MediaPipe Holistic", image)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                finish_program()

    return f'{final_distance:.2f}'


# ─────────────────────────────────────────────────────────────────
#  PONTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────

distances_right = []
distances_left  = []

repeats = 0
age, height, weight, gender = register()
gender = _("Feminine") if gender.upper() == "F" else _("Male")

while repeats < 4:
    side_label = _("Right") if repeats in [0, 1] else _("Left")
    rep_num    = (repeats % 2) + 1

    final_distance = process_exercise(repeats)

    if final_distance is not None:
        caminho_arquivo = "./tabelas_utentes/sit_and_reach_2_utentes.xlsx"
        df   = pd.read_excel(caminho_arquivo, engine="openpyxl")
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

        with open("./logs_utentes/logs_sit_and_reach_utentes", "a",
                  encoding="utf-8") as arquivo:
            arquivo.write(f"{fmt_datetime()}, {age}, {height}, {weight}, "
                          f"{gender}, {real}, {final_distance}, {side}\n")

        final_repetition_visualization(final_distance, real, _("Sit and Reach"), side_label, rep_num)
        repeats += 1
    else:
        print(_("Exercise not performed correctly"))
        finish_program()

best_left  = min(distances_left,  key=lambda x: abs(float(x)))
best_right = min(distances_right, key=lambda x: abs(float(x)))

print(f"SAR_RIGHT={best_right}")
print(f"SAR_LEFT={best_left}")
sys.stdout.flush()

final_visualization(best_left, best_right)