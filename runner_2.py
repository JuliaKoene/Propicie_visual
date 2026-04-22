import subprocess
import sys
import re
import numpy as np
import cv2
import time
import os
import gettext
from PIL import Image, ImageDraw, ImageFont

fontFile = "LiberationSansBold.ttf"

language = "en_US"
args = sys.argv
if len(args) >= 2:
    language = args[1]
lang = gettext.translation("messages", localedir="locale", languages=[language])
lang.install()

# ─────────────────────────────────────────────
#  IDENTIDADE VISUAL CAPACITA — paleta BGR
# ─────────────────────────────────────────────
C_BG        = (247, 240, 234)   # #EAF0F7
C_PRIMARY   = (143,  46,  45)   # #2D2E8F  azul escuro
C_SECONDARY = (184,  78,  60)   # #3C4EB8  azul médio
C_WHITE     = (255, 255, 255)
C_DARK_TEXT = ( 45,  46, 141)
C_LIGHT_TXT = (200, 200, 220)
C_SUCCESS   = ( 80, 180,  80)
C_ACCENT    = (184,  78,  60)   # alias de C_SECONDARY para compatibilidade

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
    text_height = bbox[3] - bbox[1]
    adjusted_y = pos[1] - text_height
    
    # Desenha o texto
    draw.text((pos[0], adjusted_y), text, font=font, fill=color_rgb)
    
    # Converte de volta para OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_text_size_utf8(text, font_size):
    font = get_font(font_size)
    # Usa imagem dummy para calcular
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def _capacita_bg(h=600, w=900):
    img = np.ones((h, w, 3), dtype=np.uint8)
    img[:] = C_BG
    return img


def _draw_capacita_header(img, title_text, w=900):
    """Cabeçalho de página com título e linhas decorativas estilo CAPACITA."""
    title_x = 60
    title_y  = 30
    tw, th = get_text_size_utf8(title_text, 42)
    img = put_text_utf8(img, title_text, (title_x + 20, title_y + th), font_size=42, color_bgr=C_PRIMARY)
    cv2.line(img, (title_x, title_y + th // 2 + 4),
             (title_x + 16, title_y + th // 2 + 4), C_PRIMARY, 3)
    cv2.line(img, (title_x + 20 + tw + 12, title_y + th // 2 + 4),
             (w - 60, title_y + th // 2 + 4), C_PRIMARY, 3)
    return img, title_y + th + 20


def _draw_capacita_box(img, x1, y1, x2, y2):
    cv2.rectangle(img, (x1, y1), (x2, y2), C_PRIMARY, 2)
    return img


# ══════════════════════════════════════════════
#  INTRO SCREEN
# ══════════════════════════════════════════════

def intro_screen():
    steps = [
        ("1", _("Sit and Reach"), _("Right Leg  x2  ->  Left Leg  x2"), C_SECONDARY),
        ("2", _("Back Scratch"),  _("Right Side x2  ->  Left Side x2"), C_PRIMARY),
    ]
    cv2.namedWindow("Assessment Protocol", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Assessment Protocol", 900, 600)
    cv2.moveWindow("Assessment Protocol", 500, 200)

    for frame_idx in range(9999):
        img = _capacita_bg()

        # IPBeja / Operador
        op_text = "IPBeja / Operador"
        tw, _ = get_text_size_utf8(op_text, 20)
        img = put_text_utf8(img, op_text, (900 - tw - 30, 44), font_size=20, color_bgr=C_DARK_TEXT)

        # Título principal CAPACITA
        title = _("Fitness Assessment  v2.0")
        tw2, th2 = get_text_size_utf8(title, 42)
        img = put_text_utf8(img, title, (60 + 20, 30 + th2), font_size=42, color_bgr=C_PRIMARY)
        cv2.line(img, (60, 30 + th2 // 2 + 4), (76, 30 + th2 // 2 + 4), C_PRIMARY, 3)
        cv2.line(img, (80 + tw2 + 12, 30 + th2 // 2 + 4), (840, 30 + th2 // 2 + 4), C_PRIMARY, 3)

        content_top = 30 + th2 + 20

        # Caixa de conteúdo
        cv2.rectangle(img, (60, content_top), (840, 540), C_PRIMARY, 2)

        # Subtítulo dentro da caixa
        sub = _("Exercise sequence for this session:")
        img = put_text_utf8(img, sub, (80, content_top + 38), font_size=20, color_bgr=C_LIGHT_TXT)

        for i, (num, name, detail, col) in enumerate(steps):
            cy = content_top + 70 + i * 130

            # Faixa lateral de cor
            cv2.rectangle(img, (60, cy), (68, cy + 100), col, -1)

            # Círculo com número
            cx_circle = 115
            cv2.circle(img, (cx_circle, cy + 50), 30, col, -1)
            img = put_text_utf8(img, num, (cx_circle - 8, cy + 64), font_size=28, color_bgr=C_WHITE)

            # Nome do exercício (maior)
            img = put_text_utf8(img, name, (165, cy + 42), font_size=30, color_bgr=C_PRIMARY)

            # Detalhe
            img = put_text_utf8(img, detail, (165, cy + 86), font_size=18, color_bgr=col)

        # Prompt piscante
        alpha = 0.4 + 0.6 * abs(np.sin(frame_idx * 0.09))
        prompt = _('Press  SPACE  to begin  |  ESC  to exit')
        ov2 = img.copy()
        pw, _ = get_text_size_utf8(prompt, 22)
        ov2 = put_text_utf8(ov2, prompt, (450 - pw // 2, 573), font_size=22, color_bgr=C_SUCCESS)
        cv2.addWeighted(ov2, alpha, img, 1 - alpha, 0, img)

        cv2.imshow("Assessment Protocol", img)
        key = cv2.waitKey(30) & 0xFF
        if key == 32:
            cv2.destroyWindow("Assessment Protocol")
            return
        elif key == 27:
            cv2.destroyAllWindows()
            sys.exit(0)


# ══════════════════════════════════════════════
#  GRAND FINALE
# ══════════════════════════════════════════════

def grand_finale(sar_right, sar_left, bs_right, bs_left):
    cv2.namedWindow("Assessment Complete", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Assessment Complete", 900, 600)
    cv2.moveWindow("Assessment Complete", 500, 200)

    while True:
        img = _capacita_bg()

        # IPBeja / Operador
        op_text = "IPBeja / Operador"
        tw, _ = get_text_size_utf8(op_text, 20)
        img = put_text_utf8(img, op_text, (900 - tw - 30, 44), font_size=20, color_bgr=C_DARK_TEXT)

        # Título
        title = _("Assessment complete") + "!"
        tw2, th2 = get_text_size_utf8(title, 42)
        img = put_text_utf8(img, title, (60 + 20, 30 + th2), font_size=42, color_bgr=C_PRIMARY)
        cv2.line(img, (60, 30 + th2 // 2 + 4), (76, 30 + th2 // 2 + 4), C_SUCCESS, 3)
        cv2.line(img, (80 + tw2 + 12, 30 + th2 // 2 + 4), (840, 30 + th2 // 2 + 4), C_SUCCESS, 3)

        content_top = 30 + th2 + 20
        cv2.rectangle(img, (60, content_top), (840, 530), C_PRIMARY, 2)

        sections = [
            (_("Sit and Reach"), [
                (f"{_('Best Right Leg')} : {sar_right} cm", C_SECONDARY),
                (f"{_('Best Left Leg')}  : {sar_left}  cm", C_SECONDARY),
            ], C_SECONDARY),
            (_("Back Scratch"), [
                (f"{_('Best Right Side')}: {bs_right} cm", C_PRIMARY),
                (f"{_('Best Left Side')} : {bs_left}  cm", C_PRIMARY),
            ], C_PRIMARY),
        ]

        for si, (title_sec, rows, col) in enumerate(sections):
            base_y = content_top + 20 + si * 200

            # Barra de título da secção
            cv2.rectangle(img, (80, base_y), (820, base_y + 38), col, -1)
            img = put_text_utf8(img, title_sec,
                                (96, base_y + 32), font_size=24, color_bgr=C_WHITE)

            for ri, (text, c) in enumerate(rows):
                row_y = base_y + 56 + ri * 60
                cv2.rectangle(img, (80, row_y), (820, row_y + 46), C_WHITE, -1)
                cv2.rectangle(img, (80, row_y), (820, row_y + 46), col, 1)
                img = put_text_utf8(img, text, (96, row_y + 40), font_size=22, color_bgr=C_DARK_TEXT)

        cv2.line(img, (60, 530), (840, 530), C_GREY, 1)
        
        prompt = _('Press  "Q"  to exit')
        pw, _ = get_text_size_utf8(prompt, 22)
        img = put_text_utf8(img, prompt, (450 - pw // 2, 575), font_size=22, color_bgr=C_DARK_TEXT)

        cv2.imshow("Assessment Complete", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


# ══════════════════════════════════════════════
#  SUBPROCESS RUNNER — reads stdout line by line
# ══════════════════════════════════════════════

def run_and_collect(script_path, keys):
    results = {k: "N/A" for k in keys}
    # Use line-buffered stdout from child
    proc = subprocess.Popen(
        [sys.executable, "-u", script_path, language],
        stdout=subprocess.PIPE,
        stderr=None,          # let stderr go to our terminal
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(line)       # echo to our own terminal
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

    intro_screen()

    # ── 1. Sit and Reach ──────────────────────
    print("\n" + "=" * 50)
    print("  " + _("Starting") + ": " + _("Sit and Reach"))
    print("=" * 50)
    sar = run_and_collect(sar_script, ["SAR_RIGHT", "SAR_LEFT"])
    print("  " + _("Sit and Reach") + " — " + _("Right") + f": {sar['SAR_RIGHT']} cm | " + _("Left") + f": {sar['SAR_LEFT']} cm")

    # ── 2. Back Scratch ───────────────────────
    print("\n" + "=" * 50)
    print("  " + _("Starting") + ": " + _("Back Scratch"))
    print("=" * 50)
    bs = run_and_collect(bs_script, ["BS_RIGHT", "BS_LEFT"])
    print("  " + _("Back Scratch") + " — " + _("Right") + f": {bs['BS_RIGHT']} cm | " + _("Left") + f": {bs['BS_LEFT']} cm")

    # ── Grand finale ──────────────────────────
    grand_finale(sar["SAR_RIGHT"], sar["SAR_LEFT"],
                 bs["BS_RIGHT"],  bs["BS_LEFT"])

    print("\n" + _("Assessment complete") + ". " + _("All data saved") + ".")