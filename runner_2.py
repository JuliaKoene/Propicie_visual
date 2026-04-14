"""
runner_v2.py
────────────
Fixed exercise sequence:
  1. Sit and Reach  — 2 reps Right, 2 reps Left
  2. Back Scratch   — 2 reps Right, 2 reps Left

Results are printed to stdout by each sub-script and read live
(no capture_output so the Kinect windows remain visible).
"""

import subprocess
import sys
import re
import numpy as np
import cv2
import time
import os

# ─────────────────────────────────────────────
#  COLOUR PALETTE  (BGR)
# ─────────────────────────────────────────────
C_BG      = (255,  255,  255)
C_ACCENT  = (0,  210, 255)
C_SUCCESS = (0,  220, 100)
C_WARN    = (0,  160, 255)
C_WHITE   = (255, 255, 255)
C_GREY    = (140, 140, 160)
C_YELLOW  = (0,  230, 230)


def _gradient_bg(h=600, w=900):
    img = np.ones((h,w,3), dtype=np.uint8) * 255
    return img


# ══════════════════════════════════════════════
#  INTRO SCREEN
# ══════════════════════════════════════════════

def intro_screen():
    steps = [
        ("1","Sit and Reach","Right Leg  x2  ->  Left Leg  x2",  C_ACCENT),
        ("2","Back Scratch", "Right Side x2  ->  Left Side x2",  C_YELLOW),
    ]
    cv2.namedWindow("Assessment Protocol", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Assessment Protocol", 900, 600)
    cv2.moveWindow("Assessment Protocol", 500, 200)

    for frame_idx in range(9999):   # loop until keypress
        img = _gradient_bg()
        cv2.rectangle(img,(0,0),(900,6),C_ACCENT,-1)
        cv2.putText(img,"Fitness Assessment  v2.0",(60,65),
                    cv2.FONT_HERSHEY_DUPLEX,1.3,(0,0,0),2,cv2.LINE_AA)
        cv2.line(img,(60,82),(840,82),C_ACCENT,1)
        cv2.putText(img,"Exercise sequence for this session:",(60,120),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75,C_GREY,1,cv2.LINE_AA)

        for i,(num,name,detail,col) in enumerate(steps):
            cy = 170 + i*140
            ov = img.copy()
            cv2.rectangle(ov,(55,cy-10),(845,cy+105),(35,35,55),-1)
            cv2.addWeighted(ov,0.85,img,0.15,0,img)
            cv2.rectangle(img,(55,cy-10),(65,cy+105),col,-1)
            cv2.circle(img,(105,cy+45),28,col,-1)
            cv2.putText(img,num,(97,cy+54),cv2.FONT_HERSHEY_DUPLEX,1.1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(img,name,(150,cy+38),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(img,detail,(150,cy+80),cv2.FONT_HERSHEY_SIMPLEX,0.72,col,1,cv2.LINE_AA)

        alpha = 0.5+0.5*abs(np.sin(frame_idx*0.08))
        prompt = 'Press  SPACE  to begin  |  ESC  to exit'
        ov2 = img.copy()
        (pw,_),_ = cv2.getTextSize(prompt,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
        cv2.putText(ov2,prompt,(900//2-pw//2,548),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,C_SUCCESS,2,cv2.LINE_AA)
        cv2.addWeighted(ov2,alpha,img,1-alpha,0,img)

        cv2.imshow("Assessment Protocol",img)
        key=cv2.waitKey(30)&0xFF
        if key==32:
            cv2.destroyWindow("Assessment Protocol"); return
        elif key==27:
            cv2.destroyAllWindows(); sys.exit(0)


# ══════════════════════════════════════════════
#  GRAND FINALE
# ══════════════════════════════════════════════

def grand_finale(sar_right, sar_left, bs_right, bs_left):
    cv2.namedWindow("Assessment Complete", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Assessment Complete", 900, 600)
    cv2.moveWindow("Assessment Complete", 500, 200)
    while True:
        img = _gradient_bg()
        cv2.rectangle(img,(0,0),(900,6),C_SUCCESS,-1)
        cv2.putText(img,"Assessment Complete!",(180,68),
                    cv2.FONT_HERSHEY_DUPLEX,1.4,(0,0,0),2,cv2.LINE_AA)
        cv2.line(img,(60,88),(840,88),C_SUCCESS,1)

        sections = [
            ("Sit and Reach",[
                (f"Best Right Leg : {sar_right} cm", C_ACCENT),
                (f"Best Left Leg  : {sar_left}  cm", C_ACCENT),
            ]),
            ("Back Scratch",[
                (f"Best Right Side: {bs_right} cm", C_YELLOW),
                (f"Best Left Side : {bs_left}  cm", C_YELLOW),
            ]),
        ]
        for si,(title,rows) in enumerate(sections):
            base_y=140+si*190
            col = C_ACCENT if si==0 else C_YELLOW
            cv2.putText(img,title,(70,base_y),cv2.FONT_HERSHEY_DUPLEX,1.0,col,2,cv2.LINE_AA)
            cv2.line(img,(70,base_y+10),(500,base_y+10),C_GREY,1)
            for ri,(text,c) in enumerate(rows):
                cv2.putText(img,text,(90,base_y+55+ri*50),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,c,2,cv2.LINE_AA)

        cv2.line(img,(60,530),(840,530),C_GREY,1)
        prompt='Press  "Q"  to exit'
        (pw,_),_=cv2.getTextSize(prompt,cv2.FONT_HERSHEY_SIMPLEX,0.8,1)
        cv2.putText(img,prompt,(900//2-pw//2,570),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,C_YELLOW,1,cv2.LINE_AA)
        cv2.imshow("Assessment Complete",img)
        if cv2.waitKey(1)&0xFF==ord('q'):
            cv2.destroyAllWindows(); break


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
        [sys.executable, "-u", script_path],
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
    print("\n" + "="*50)
    print("  Starting: Sit and Reach")
    print("="*50)
    sar = run_and_collect(sar_script, ["SAR_RIGHT","SAR_LEFT"])
    print(f"  Sit and Reach — Right: {sar['SAR_RIGHT']} cm | Left: {sar['SAR_LEFT']} cm")

    # ── 2. Back Scratch ───────────────────────
    print("\n" + "="*50)
    print("  Starting: Back Scratch")
    print("="*50)
    bs = run_and_collect(bs_script, ["BS_RIGHT","BS_LEFT"])
    print(f"  Back Scratch — Right: {bs['BS_RIGHT']} cm | Left: {bs['BS_LEFT']} cm")

    # ── Grand finale ──────────────────────────
    grand_finale(sar["SAR_RIGHT"], sar["SAR_LEFT"],
                 bs["BS_RIGHT"],   bs["BS_LEFT"])

    print("\nAssessment complete. All data saved.")