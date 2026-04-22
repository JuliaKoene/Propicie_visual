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
#  COLOUR PALETTE  (BGR for OpenCV, RGB for PIL)
# ─────────────────────────────────────────────
C_BG      = (255, 255, 255)
C_ACCENT  = (0, 210, 255)
C_SUCCESS = (0, 220, 100)
C_WARN    = (0, 160, 255)
C_WHITE   = (255, 255, 255)
C_GREY    = (140, 140, 160)
C_YELLOW  = (0, 230, 230)

# ─────────────────────────────────────────────
#  FONT CACHE (carrega uma vez por tamanho)
# ─────────────────────────────────────────────
_font_cache = {}

def get_font(size):
    """Retorna fonte TrueType do cache ou carrega se necessário."""
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(fontFile, size)
        except IOError:
            # Fallback para fonte padrão se o arquivo não existir
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

# ─────────────────────────────────────────────
#  FUNÇÃO AUXILIAR: putText com suporte UTF-8
# ─────────────────────────────────────────────
def put_text_utf8(img, text, pos, font_size, color_bgr, thickness=1):
    """
    Desenha texto UTF-8 em uma imagem OpenCV usando PIL.
    
    Args:
        img: Imagem OpenCV (numpy array BGR)
        text: String a ser desenhada (suporta UTF-8)
        pos: Tupla (x, y) - posição do canto inferior esquerdo do texto
        font_size: Tamanho da fonte em pixels
        color_bgr: Cor em formato BGR (OpenCV)
        thickness: Ignorado, mantido para compatibilidade de assinatura
    
    Returns:
        Imagem modificada
    """
    # Converte BGR (OpenCV) para RGB (PIL)
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
    """
    Retorna o tamanho do texto em pixels.
    
    Args:
        text: String a ser medida
        font_size: Tamanho da fonte
    
    Returns:
        Tupla (largura, altura)
    """
    font = get_font(font_size)
    # Usa imagem dummy para calcular
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def _gradient_bg(h=600, w=900):
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    return img


# ══════════════════════════════════════════════
#  INTRO SCREEN
# ══════════════════════════════════════════════

def intro_screen():
    steps = [
        ("1", _("Sit and Reach"), _("Right Leg  x2  ->  Left Leg  x2"), C_ACCENT),
        ("2", _("Back Scratch"), _("Right Side x2  ->  Left Side x2"), C_YELLOW),
    ]
    cv2.namedWindow("Assessment Protocol", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Assessment Protocol", 900, 600)
    cv2.moveWindow("Assessment Protocol", 500, 200)

    for frame_idx in range(9999):   # loop until keypress
        img = _gradient_bg()
        cv2.rectangle(img, (0, 0), (900, 6), C_ACCENT, -1)
        
        # Título principal
        img = put_text_utf8(img, _("Fitness Assessment  v2.0"), (60, 65),
                           font_size=36, color_bgr=(0, 0, 0))
        
        cv2.line(img, (60, 82), (840, 82), C_ACCENT, 1)
        
        # Subtítulo
        img = put_text_utf8(img, _("Exercise sequence for this session:"), (60, 120),
                           font_size=22, color_bgr=C_GREY)

        for i, (num, name, detail, col) in enumerate(steps):
            cy = 170 + i * 140
            ov = img.copy()
            cv2.rectangle(ov, (55, cy - 10), (845, cy + 105), (35, 35, 55), -1)
            cv2.addWeighted(ov, 0.85, img, 0.15, 0, img)
            cv2.rectangle(img, (55, cy - 10), (65, cy + 105), col, -1)
            cv2.circle(img, (105, cy + 45), 28, col, -1)
            
            # Número no círculo
            img = put_text_utf8(img, num, (97, cy + 54),
                               font_size=30, color_bgr=(255, 255, 255))
            
            # Nome do exercício
            img = put_text_utf8(img, name, (150, cy + 38),
                               font_size=28, color_bgr=(255, 255, 255))
            
            # Detalhe do exercício
            img = put_text_utf8(img, detail, (150, cy + 80),
                               font_size=20, color_bgr=col)

        alpha = 0.5 + 0.5 * abs(np.sin(frame_idx * 0.08))
        prompt = _('Press  SPACE  to begin  |  ESC  to exit')
        ov2 = img.copy()
        
        pw, dm = get_text_size_utf8(prompt, font_size=22)
        ov2 = put_text_utf8(ov2, prompt, (900 // 2 - pw // 2, 548),
                           font_size=22, color_bgr=C_SUCCESS)
        
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
        img = _gradient_bg()
        cv2.rectangle(img, (0, 0), (900, 6), C_SUCCESS, -1)
        
        # Título
        img = put_text_utf8(img, (_("Assessment complete") + "!"), (180, 68),
                           font_size=38, color_bgr=(0, 0, 0))
        
        cv2.line(img, (60, 88), (840, 88), C_SUCCESS, 1)

        sections = [
            (_("Sit and Reach"), [
                (f"{_("Best Right Leg")} : {sar_right} cm", C_ACCENT),
                (f"{_("Best Left Leg")}  : {sar_left}  cm", C_ACCENT),
            ]),
            (_("Back Scratch"), [
                (f"{_("Best Right Side")}: {bs_right} cm", C_YELLOW),
                (f"{_("Best Left Side")} : {bs_left}  cm", C_YELLOW),
            ]),
        ]
        
        for si, (title, rows) in enumerate(sections):
            base_y = 140 + si * 190
            col = C_ACCENT if si == 0 else C_YELLOW
            
            # Título da seção
            img = put_text_utf8(img, title, (70, base_y),
                               font_size=28, color_bgr=col)
            
            cv2.line(img, (70, base_y + 10), (500, base_y + 10), C_GREY, 1)
            
            for ri, (text, c) in enumerate(rows):
                img = put_text_utf8(img, text, (90, base_y + 55 + ri * 50),
                                   font_size=24, color_bgr=c)

        cv2.line(img, (60, 530), (840, 530), C_GREY, 1)
        
        prompt = _('Press  "Q"  to exit')
        pw, dm = get_text_size_utf8(prompt, font_size=22)
        img = put_text_utf8(img, prompt, (900 // 2 - pw // 2, 570),
                           font_size=22, color_bgr=C_YELLOW)
        
        cv2.imshow("Assessment Complete", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


# ══════════════════════════════════════════════
#  SUBPROCESS RUNNER — reads stdout line by line
# ══════════════════════════════════════════════

def run_and_collect(script_path, keys):
    """
    Run script_path as a subprocess.
    Stdout is NOT captured globally; we read it line-by-line so the
    Kinect windows still open normally.  Values printed as KEY=value
    are collected and returned as a dict.
    """
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
    bs_script = os.path.join(base, "Back-Scratch", "back_scratch_julia.py")

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
                 bs["BS_RIGHT"], bs["BS_LEFT"])

    print("\n" + _("Assessment complete") + ". " + _("All data saved") + ".")
