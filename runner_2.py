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


#  COLOUR PALETTE  (BGR)
C_BG      = (255,  255,  255)
C_ACCENT  = (0,  210, 255)
C_SUCCESS = (0,  220, 100)
C_WARN    = (0,  160, 255)
C_WHITE   = (255, 255, 255)
C_GREY    = (140, 140, 160)
C_YELLOW  = (0,  230, 230)

# Screen resolution constants
SCREEN_WIDTH  = 1920
SCREEN_HEIGHT = 1080


def _gradient_bg(h=SCREEN_HEIGHT, w=SCREEN_WIDTH):
    img = np.ones((h,w,3), dtype=np.uint8) * 255
    return img
#  INTRO SCREEN
def intro_screen():
    steps = [
        ("1","Sit and Reach","Right Leg  x2  ->  Left Leg  x2",  C_ACCENT),
        ("2","Back Scratch", "Right Side x2  ->  Left Side x2",  C_YELLOW),
    ]
    cv2.namedWindow("CAPACITA Project - Assessment Protocol", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("CAPACITA Project - Assessment Protocol", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for frame_idx in range(9999):   # loop until keypress
        img = _gradient_bg()
        cv2.rectangle(img,(0,0),(SCREEN_WIDTH,8),C_ACCENT,-1)
        (tw, _), _ = cv2.getTextSize("Fitness Assessment  v2.0", cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)
        cv2.putText(img,"Fitness Assessment  v2.0",((SCREEN_WIDTH-tw)//2,120),
                    cv2.FONT_HERSHEY_DUPLEX,2.0,(0,0,0),3,cv2.LINE_AA)
        cv2.line(img,(100,160),(SCREEN_WIDTH-100,160),C_ACCENT,2)
        cv2.putText(img,"Exercise sequence for this session:",(150,220),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,C_GREY,2,cv2.LINE_AA)

        for i,(num,name,detail,col) in enumerate(steps):
            cy = 350 + i*300
            ov = img.copy()
            cv2.rectangle(ov,(100,cy-20),(SCREEN_WIDTH-100,cy+180),(35,35,55),-1)
            cv2.addWeighted(ov,0.85,img,0.15,0,img)
            cv2.rectangle(img,(100,cy-20),(120,cy+180),col,-1)
            cv2.circle(img,(180,cy+80),50,col,-1)
            cv2.putText(img,num,(160,cy+100),cv2.FONT_HERSHEY_DUPLEX,2.0,(255,255,255),3,cv2.LINE_AA)
            cv2.putText(img,name,(280,cy+70),cv2.FONT_HERSHEY_DUPLEX,1.8,(255,255,255),3,cv2.LINE_AA)
            cv2.putText(img,detail,(280,cy+130),cv2.FONT_HERSHEY_SIMPLEX,1.3,col,2,cv2.LINE_AA)

        alpha = 0.5+0.5*abs(np.sin(frame_idx*0.08))
        prompt = 'Press  SPACE  to begin  |  ESC  to exit'
        ov2 = img.copy()
        (pw,_),_ = cv2.getTextSize(prompt,cv2.FONT_HERSHEY_SIMPLEX,1.2,2)
        cv2.putText(ov2,prompt,(SCREEN_WIDTH//2-pw//2,SCREEN_HEIGHT-100),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,C_SUCCESS,2,cv2.LINE_AA)
        cv2.addWeighted(ov2,alpha,img,1-alpha,0,img)

        cv2.imshow("CAPACITA Project - Assessment Protocol",img)
        key=cv2.waitKey(30)&0xFF
        if key==32:
            cv2.destroyWindow("CAPACITA Project - Assessment Protocol"); return
        elif key==27:
            cv2.destroyAllWindows(); sys.exit(0)

#  FINALE
def grand_finale(sar_right, sar_left, bs_right, bs_left):
    cv2.namedWindow("CAPACITA Project - Assessment Complete", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("CAPACITA Project - Assessment Complete", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        img = _gradient_bg()
        cv2.rectangle(img,(0,0),(SCREEN_WIDTH,8),C_SUCCESS,-1)
        (tw, _), _ = cv2.getTextSize("Assessment Complete!", cv2.FONT_HERSHEY_DUPLEX, 2.5, 3)
        cv2.putText(img,"Assessment Complete!",((SCREEN_WIDTH-tw)//2,140),
                    cv2.FONT_HERSHEY_DUPLEX,2.5,(0,0,0),3,cv2.LINE_AA)
        cv2.line(img,(100,180),(SCREEN_WIDTH-100,180),C_SUCCESS,2)

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
            base_y=300+si*350
            col = C_ACCENT if si==0 else C_YELLOW
            cv2.putText(img,title,(200,base_y),cv2.FONT_HERSHEY_DUPLEX,1.8,col,3,cv2.LINE_AA)
            cv2.line(img,(200,base_y+30),(800,base_y+30),C_GREY,2)
            for ri,(text,c) in enumerate(rows):
                cv2.putText(img,text,(300,base_y+100+ri*80),
                            cv2.FONT_HERSHEY_SIMPLEX,1.5,c,2,cv2.LINE_AA)

        cv2.line(img,(100,SCREEN_HEIGHT-200),(SCREEN_WIDTH-100,SCREEN_HEIGHT-200),C_GREY,2)
        prompt='Press  "Q"  to exit'
        (pw,_),_=cv2.getTextSize(prompt,cv2.FONT_HERSHEY_SIMPLEX,1.3,2)
        cv2.putText(img,prompt,(SCREEN_WIDTH//2-pw//2,SCREEN_HEIGHT-100),
                    cv2.FONT_HERSHEY_SIMPLEX,1.3,C_YELLOW,2,cv2.LINE_AA)
        cv2.imshow("Assessment Complete",img)
        if cv2.waitKey(1)&0xFF==ord('q'):
            cv2.destroyAllWindows(); break
#  SUBPROCESS RUNNER — reads stdout line by line
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
#  MAIN
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