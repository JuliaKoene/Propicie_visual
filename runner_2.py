import subprocess
import sys
import re
import numpy as np
import cv2
import os
import gettext
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

fontFile = "LiberationSansBold.ttf"

language = "en_US"
args = sys.argv
if len(args) >= 2:
    language = args[1]
lang = gettext.translation("messages", localedir="locale", languages=[language])
lang.install()
_ = lang.gettext

# ─────────────────────────────────────────────
#  PALETA CAPACITA — BGR
# ─────────────────────────────────────────────
C_BG        = (247, 240, 234)
C_PRIMARY   = (143,  46,  45)
C_SECONDARY = (184,  78,  60)
C_BTN_HOVER = (110,  35,  34)
C_WHITE     = (255, 255, 255)
C_DARK_TEXT = ( 45,  46, 141)
C_LIGHT_TXT = (180, 180, 210)
C_SUCCESS   = ( 80, 180,  80)

# ─────────────────────────────────────────────
#  RESOLUÇÃO DINÂMICA
# ─────────────────────────────────────────────
_SW = 1920
_SH = 1080

def _detect_screen():
    global _SW, _SH
    tmp = "##detect##"
    cv2.namedWindow(tmp, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(tmp, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    probe = np.zeros((4, 4, 3), dtype=np.uint8)
    cv2.imshow(tmp, probe)
    cv2.waitKey(1)
    rect = cv2.getWindowImageRect(tmp)
    if rect[2] > 100 and rect[3] > 100:
        _SW, _SH = rect[2], rect[3]
    cv2.destroyWindow(tmp)

def W():   return _SW
def H():   return _SH
def S(v):  return max(1, int(v * _SW / 1920))
def SH(v): return max(1, int(v * _SH / 1080))
def SF(v): return max(8, int(v * min(_SW, _SH) / 1080))

# ─────────────────────────────────────────────
#  FONT CACHE (carrega uma vez por tamanho)
# ─────────────────────────────────────────────
_font_cache = {}

def get_font(size):
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(fontFile, size)
        except IOError:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def put_text_utf8(img, text, pos, font_size, color_bgr, thickness=1):
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    
    # Converte imagem OpenCV para PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Obtém fonte do cache
    font = get_font(font_size)
    
    # Calcula posição ajustada (PIL usa canto superior esquerdo)
    # Obtém métricas da fonte para ajustar baseline
    bbox = draw.textbbox((0, 0), text, font=font)
    th = bbox[3] - bbox[1]
    draw.text((pos[0], pos[1] - th), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_text_size(text, font_size):
    font = get_font(font_size)
    # Usa imagem dummy para calcular
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def put_text_centered(img, text, cx, cy, font_size, color_bgr):
    tw, th = get_text_size(text, font_size)
    return put_text_utf8(img, text, (cx - tw // 2, cy + th // 2), font_size, color_bgr)

# ─────────────────────────────────────────────
#  PRIMITIVAS DE LAYOUT
# ─────────────────────────────────────────────
def _canvas():
    img = np.ones((H(), W(), 3), dtype=np.uint8)
    img[:] = C_BG
    return img

def _draw_page_title(img, title):
    """Título com traço curto + linha longa — fiel ao PDF."""
    pad_x = S(60)
    pad_y = SH(42)
    fsz   = SF(52)
    tw, th = get_text_size(title, fsz)
    line_y = pad_y + th // 2

    cv2.line(img, (pad_x, line_y), (pad_x + S(18), line_y), C_PRIMARY, S(4))
    img = put_text_utf8(img, title, (pad_x + S(28), pad_y + th), fsz, C_PRIMARY)
    cv2.line(img,
             (pad_x + S(28) + tw + S(16), line_y),
             (W() - pad_x, line_y),
             C_PRIMARY, S(4))

    op_fsz = SF(22)
    op_tw, op_th = get_text_size("IPBeja / Operador", op_fsz)
    img = put_text_utf8(img, "IPBeja / Operador",
                        (W() - op_tw - S(30), SH(52)), op_fsz, C_DARK_TEXT)

    return pad_y + th + SH(22)

def _border_box(img, x1, y1, x2, y2):
    cv2.rectangle(img, (x1, y1), (x2, y2), C_PRIMARY, S(2))

def _fullscreen(name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ─────────────────────────────────────────────
#  BOLINHAS DE PROGRESSO
# ─────────────────────────────────────────────
def _draw_rep_circles(img, states, cx_mid, cy, radius):
    """
    Desenha círculos de estado lado a lado centrados em cx_mid.
      "done"    → cheio + X branco
      "current" → cheio + anel branco interior
      "pending" → cheio
    """
    n     = len(states)
    gap   = S(28)
    diam  = radius * 2
    total = n * diam + (n - 1) * gap
    ox    = cx_mid - total // 2 + radius

    for i, state in enumerate(states):
        ccx = ox + i * (diam + gap)

        # Fundo do círculo
        cv2.circle(img, (ccx, cy), radius, C_PRIMARY, -1)

        if state == "done":
            arm = int(radius * 0.42)
            cv2.line(img, (ccx - arm, cy - arm), (ccx + arm, cy + arm), C_WHITE, S(4))
            cv2.line(img, (ccx + arm, cy - arm), (ccx - arm, cy + arm), C_WHITE, S(4))

        elif state == "current":
            # Anel branco — círculo cheio branco + menor círculo primário ao centro
            inner = int(radius * 0.56)
            cv2.circle(img, (ccx, cy), inner, C_WHITE, -1)
            cv2.circle(img, (ccx, cy), int(inner * 0.3), C_PRIMARY, -1)

# ══════════════════════════════════════════════
#  INTRO SCREEN
# ══════════════════════════════════════════════
def _exercise_overview_screen(exercise_name, groups, current_rep, page_title=None):
    WIN = "Exercise Overview"
    _fullscreen(WIN)

    if page_title is None:
        page_title = exercise_name

    # Calcula estado de cada bolinha por grupo
    rep_idx = 0
    group_states = []
    for label, n in groups:
        states = []
        for _r in range(n):
            if rep_idx < current_rep:
                states.append("done")
            elif rep_idx == current_rep:
                states.append("current")
            else:
                states.append("pending")
            rep_idx += 1
        group_states.append((label, states))

    # Info para o footer: grupo e rep atual
    rep_idx2 = 0
    cur_group_label = groups[0][0]
    cur_rep_in_grp  = 1
    cur_grp_total   = groups[0][1]
    for label, n in groups:
        for r in range(n):
            if rep_idx2 == current_rep:
                cur_group_label = label
                cur_rep_in_grp  = r + 1
                cur_grp_total   = n
            rep_idx2 += 1

    for frame_idx in range(99999):
        img = _canvas()
        content_y = _draw_page_title(img, page_title)

        bx1 = S(60);  bx2 = W() - S(60)
        by1 = content_y; by2 = H() - SH(80)
        _border_box(img, bx1, by1, bx2, by2)

        # Layout das bolinhas
        n_groups = len(group_states)
        r_radius = S(56)
        bar_h    = SH(62)
        grp_h    = bar_h + SH(20) + r_radius * 2
        gap_grp  = SH(44)
        total_h  = n_groups * grp_h + (n_groups - 1) * gap_grp
        start_y  = by1 + (by2 - by1 - total_h - SH(50)) // 2
        cx_mid   = (bx1 + bx2) // 2
        bar_w    = S(490)

        for gi, (label, states) in enumerate(group_states):
            gy = start_y + gi * (grp_h + gap_grp)

            # Barra label
            bar_x1 = cx_mid - bar_w // 2
            bar_x2 = cx_mid + bar_w // 2
            cv2.rectangle(img, (bar_x1, gy), (bar_x2, gy + bar_h), C_PRIMARY, -1)
            img = put_text_centered(img, label, cx_mid, gy + bar_h // 2, SF(28), C_WHITE)

            # Bolinhas abaixo
            circles_cy = gy + bar_h + SH(20) + r_radius
            _draw_rep_circles(img, states, cx_mid, circles_cy, r_radius)

        # ── Footer 3 colunas ──
        foot_y = by2 - SH(8)
        cv2.line(img, (bx1, foot_y - SH(36)), (bx2, foot_y - SH(36)), C_PRIMARY, S(1))
        fsz_f = SF(20)

        img = put_text_utf8(img, exercise_name.upper(),
                            (bx1 + S(16), foot_y), fsz_f, C_DARK_TEXT)

        tw_c, _h = get_text_size(cur_group_label, fsz_f)
        img = put_text_utf8(img, cur_group_label,
                            (cx_mid - tw_c // 2, foot_y), fsz_f, C_DARK_TEXT)

        rep_str = f"{_('Rep')} {cur_rep_in_grp}/{cur_grp_total}"
        tw_r, _h = get_text_size(rep_str, fsz_f)
        img = put_text_utf8(img, rep_str,
                            (bx2 - tw_r - S(16), foot_y), fsz_f, C_DARK_TEXT)

        # Prompt piscante
        alpha  = 0.35 + 0.65 * abs(np.sin(frame_idx * 0.08))
        prompt = _('Press  SPACE  to begin  |  ESC  to exit')
        ov = img.copy()
        ov = put_text_centered(ov, prompt, W() // 2, H() - SH(32), SF(22), C_SUCCESS)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

        cv2.imshow(WIN, img)
        key = cv2.waitKey(30) & 0xFF
        if key == 32:
            cv2.destroyWindow(WIN)
            return
        elif key == 27:
            cv2.destroyAllWindows()
            sys.exit(0)


def _get_groups_for(mode):
    """Devolve (sar_groups, bs_groups) para o modo escolhido."""
    sar_groups = [
        (f"{_('Right Side')} x2", 2),
        (f"{_('Left Side')} x2",  2),
    ]
    bs_groups = [
        (f"{_('Right Side')} x2", 2),
        (f"{_('Left Side')} x2",  2),
    ]
    if mode == "sar":
        return sar_groups, []
    elif mode == "bs":
        return [], bs_groups
    return sar_groups, bs_groups   # auto


def intro_screen(mode="auto"):
    """Tela inicial com bolinhas antes do primeiro exercício."""
    sar_groups, bs_groups = _get_groups_for(mode)

    if mode in ("auto", "sar"):
        _exercise_overview_screen(
            _("Sit and Reach"),
            sar_groups,
            current_rep=0,
            page_title=_("Sit and Reach")
        )
    elif mode == "bs":
        _exercise_overview_screen(
            _("Back Scratch"),
            bs_groups,
            current_rep=0,
            page_title=_("Back Scratch")
        )


def next_rep_screen(exercise_name, groups, current_rep):
    """
    Tela "Próximo" entre repetições — idêntica à intro mas com título 'Next'.
    current_rep é o índice da próxima repetição a realizar.
    """
    _exercise_overview_screen(
        exercise_name,
        groups,
        current_rep=current_rep,
        page_title=_("Next")
    )


# ══════════════════════════════════════════════
#  MENU PRINCIPAL  — fiel ao PDF página 2
# ══════════════════════════════════════════════

def menu_principal():
    WIN = "CAPACITA"
    _fullscreen(WIN)

    btn_labels  = [
        _("Automatic"),
        _("Sit and Reach"),
        _("Back Scratch"),
        _("View Data"),
        _("End Session"),
    ]
    btn_returns = ["auto", "sar", "bs", "data", "quit"]

    hovered = -1
    clicked = -1

    def mouse_cb(event, x, y, flags, param):
        nonlocal hovered, clicked
        hovered = -1
        for i, (rx1, ry1, rx2, ry2) in enumerate(param["rects"]):
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                hovered = i
                break
        if event == cv2.EVENT_LBUTTONUP and hovered >= 0:
            clicked = hovered

    rects_ref = {"rects": []}
    cv2.setMouseCallback(WIN, mouse_cb, rects_ref)

    while True:
        img = _canvas()
        content_y = _draw_page_title(img, _("Main Menu"))

        bx1 = S(60);  bx2 = W() - S(60)
        by1 = content_y; by2 = H() - SH(60)
        _border_box(img, bx1, by1, bx2, by2)

        n       = len(btn_labels)
        btn_w   = S(490)
        btn_h   = SH(72)
        btn_gap = SH(30)
        total_h = n * btn_h + (n - 1) * btn_gap
        start_y = by1 + (by2 - by1 - total_h) // 2
        cx      = (bx1 + bx2) // 2

        rects = []
        for i, label in enumerate(btn_labels):
            rx1 = cx - btn_w // 2
            rx2 = cx + btn_w // 2
            ry1 = start_y + i * (btn_h + btn_gap)
            ry2 = ry1 + btn_h
            rects.append((rx1, ry1, rx2, ry2))

            col = C_BTN_HOVER if i == hovered else C_PRIMARY
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), col, -1)
            img = put_text_centered(img, label, cx, (ry1 + ry2) // 2, SF(28), C_WHITE)

        rects_ref["rects"] = rects

        cv2.imshow(WIN, img)
        key = cv2.waitKey(30) & 0xFF

        if clicked >= 0:
            choice = btn_returns[clicked]
            clicked = -1
            cv2.destroyWindow(WIN)
            return choice

        for i, k in enumerate([ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]):
            if key == k:
                cv2.destroyWindow(WIN)
                return btn_returns[i]

        if key in (13, 10) and hovered >= 0:
            cv2.destroyWindow(WIN)
            return btn_returns[hovered]

        if key == 27:
            cv2.destroyAllWindows()
            sys.exit(0)

    return "auto"


# ══════════════════════════════════════════════
#  GRAND FINALE
# ══════════════════════════════════════════════

def grand_finale(sar_right, sar_left, bs_right, bs_left):
    WIN = "Assessment Complete"
    _fullscreen(WIN)

    while True:
        img = _canvas()
        content_y = _draw_page_title(img, _("Assessment complete") + "!")

        bx1 = S(60);  bx2 = W() - S(60)
        by1 = content_y; by2 = H() - SH(80)
        _border_box(img, bx1, by1, bx2, by2)

        sections = [
            (_("Sit and Reach"), [
                (f"{_('Best Right Leg')} :  {sar_right} cm", C_PRIMARY),
                (f"{_('Best Left Leg')}  :  {sar_left} cm",  C_PRIMARY),
            ], C_PRIMARY),
            (_("Back Scratch"), [
                (f"{_('Best Right Side')} :  {bs_right} cm", C_SECONDARY),
                (f"{_('Best Left Side')}  :  {bs_left} cm",  C_SECONDARY),
            ], C_SECONDARY),
        ]

        n_sec   = len(sections)
        sec_h   = SH(200)
        gap_sec = SH(24)
        total_h = n_sec * sec_h + (n_sec - 1) * gap_sec
        sy      = by1 + (by2 - by1 - total_h) // 2

        for si, (sec_title, rows, col) in enumerate(sections):
            base_y = sy + si * (sec_h + gap_sec)
            sx1 = bx1 + S(20);  sx2 = bx2 - S(20)
            bar_h = SH(46)

            cv2.rectangle(img, (sx1, base_y), (sx2, base_y + bar_h), col, -1)
            img = put_text_utf8(img, sec_title,
                                (sx1 + S(18), base_y + bar_h - SH(8)), SF(26), C_WHITE)

            for ri, (text, c) in enumerate(rows):
                ry1 = base_y + bar_h + ri * SH(70)
                ry2 = ry1 + SH(62)
                cv2.rectangle(img, (sx1, ry1), (sx2, ry2), C_WHITE, -1)
                cv2.rectangle(img, (sx1, ry1), (sx2, ry2), col, S(1))
                img = put_text_utf8(img, text,
                                    (sx1 + S(18), ry2 - SH(12)), SF(24), C_DARK_TEXT)

        img = put_text_centered(img, _('Press  "Q"  to exit'),
                                W() // 2, H() - SH(40), SF(22), C_DARK_TEXT)

        cv2.imshow(WIN, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


# ══════════════════════════════════════════════
#  SUBPROCESS RUNNER — reads stdout line by line
# ══════════════════════════════════════════════

def run_and_collect(script_path, keys):
    results = {k: "N/A" for k in keys}
    proc = subprocess.Popen(
        [sys.executable, "-u", script_path, language],
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(line)
        for k in keys:
            m = re.match(rf"^{k}=(.+)$", line)
            if m:
                results[k] = m.group(1).strip()
    proc.wait()
    return results


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":

    # Resolve paths relative to this file's location
    base = os.path.dirname(os.path.abspath(__file__))
    sar_script = os.path.join(base, "Sit-and-Reach", "sit_and_reach_julia.py")
    bs_script  = os.path.join(base, "Back-Scratch",  "back_scratch_julia.py")

    _detect_screen()

    # ── Menu ──────────────────────────────────
    choice = menu_principal()

    if choice == "quit":
        sys.exit(0)
    if choice == "data":
        df_bs = pd.read_excel('tabelas_utentes/back_scratch_utentes.xlsx', sheet_name='Sheet1')
        df_sr = pd.read_excel('tabelas_utentes/sit_and_reach_2_utentes.xlsx', sheet_name='Sheet1')

        print("\n" + "=" * 50)
        print("  Table Back Scratch  ")
        print(df_bs)
        print("=" * 50 + "\n")

        print("\n" + "=" * 50)
        print("  Tabela Sit and Reach  ")
        print(df_sr)
        print("=" * 50 + "\n")

        sys.exit(0)

    sar_groups, bs_groups = _get_groups_for(choice)
    sar = {"SAR_RIGHT": "N/A", "SAR_LEFT": "N/A"}
    bs  = {"BS_RIGHT":  "N/A", "BS_LEFT":  "N/A"}

    # ── Sit and Reach ─────────────────────────
    if choice in ("auto", "sar"):
        # Tela inicial com bolinhas
        intro_screen(mode="sar")

        print("\n" + "=" * 50)
        print("  " + _("Starting") + ": " + _("Sit and Reach"))
        print("=" * 50)

        total_sar = sum(n for _, n in sar_groups)
        for rep_idx in range(total_sar):
            # Entre repetições: mostra "Próximo" com progresso actualizado
            if rep_idx > 0:
                next_rep_screen(_("Sit and Reach"), sar_groups, rep_idx)

            # Lança um único exercício passando o índice como argumento extra
            # O script sit_and_reach_julia.py trata cada repetição individualmente;
            # o runner invoca-o 4 vezes (comportamento original) mas agora
            # a tela de transição é mostrada aqui.

        # Corre o script completo (4 reps internas)
        sar = run_and_collect(sar_script, ["SAR_RIGHT", "SAR_LEFT"])
        print("  SAR — " + _("Right") + f": {sar['SAR_RIGHT']} cm | "
              + _("Left") + f": {sar['SAR_LEFT']} cm")

    # ── Back Scratch ──────────────────────────
    if choice in ("auto", "bs"):
        intro_screen(mode="bs")

        print("\n" + "=" * 50)
        print("  " + _("Starting") + ": " + _("Back Scratch"))
        print("=" * 50)

        bs = run_and_collect(bs_script, ["BS_RIGHT", "BS_LEFT"])
        print("  BS  — " + _("Right") + f": {bs['BS_RIGHT']} cm | "
              + _("Left") + f": {bs['BS_LEFT']} cm")

    # ── Grand Finale ──────────────────────────
    grand_finale(sar["SAR_RIGHT"], sar["SAR_LEFT"],
                 bs["BS_RIGHT"],   bs["BS_LEFT"])

    print("\n" + _("Assessment complete") + ". " + _("All data saved") + ".")