from pykinect2 import PyKinectRuntime, PyKinectV2
import mediapipe as mp
import datetime as dt  
import pandas as pd
import numpy as np
import time
import cv2

# Approximate ratio of pixels to cm at 1 meter distance
PIXEL_TO_CM_RATIO = 0.625

# variable initialization
POSE_NO_HELD_DURATION = 1.5
POSE_HELD_DURATION = 3
DISTANCE_CHECK = 33
AVERAGE_OVER = 5
ERROR = 1.91


# Kinect initialization
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

# Initialize MediaPipe Holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Media Pipe Holistic initialization
def finish_program():
    cv2.destroyAllWindows()
    kinect.close()
    exit()

# Function to calculate Euclidean distance in 2D (x, y only)
def calculate_distance_2d(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to calculate average distance
def average_distance(distances):
    return sum(distances) / len(distances)

# Function to process kinect frames
def process_frame(kinect):
    frame = kinect.get_last_color_frame()
    frame = frame.reshape((1080, 1920, 4))  # BGRA
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    rgb_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = holistic.process(rgb_frame)
    rgb_frame.flags.writeable = True
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), results, frame

# Function to show the performance screen for that attempt
def final_repetition_visualization(distance,real_distance):

    final_repetition_frame = np.zeros((500, 800, 3), dtype=np.uint8)

    cv2.putText(final_repetition_frame, f'Repetition Completed', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(final_repetition_frame, f"Distance between both hands: {distance} cm", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(final_repetition_frame, f'Real Distance: {real_distance} centimeters', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(final_repetition_frame,f'Press "c" to continue or "q" to finish the exercise',(50,400),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,0),2)
    cv2.imshow("Repetition Results", final_repetition_frame)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            finish_program()
        elif key == ord('c'):
            cv2.destroyWindow("Repetition Results")
            break  

# Function to show the final display
def final_visualization(left,right):
    final_frame = np.zeros((500,800,3),dtype=np.uint8)

    cv2.putText(final_frame,f'Exercise completed',(200,100),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(final_frame, f'Better result of the right side: {right} cm', (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(final_frame, f'Better result of the left side: {left} cm', (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(final_frame,f'Press "q" to finish the exercise',(200,400),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,0),2)

    cv2.imshow("Final results",final_frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to exit
            finish_program() 

# Draw hands landmarks
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    
# Draw only middle finger landmark
def draw_middle_finger_only(image, hand_landmarks, color=(0, 255, 0)):
    middle_indices = [9, 10, 11, 12]
    h, w, _ = image.shape

    for i in middle_indices:
        x = int(hand_landmarks.landmark[i].x * w)
        y = int(hand_landmarks.landmark[i].y * h)
        cv2.circle(image, (x, y), 5, color, -1)

    for i in range(len(middle_indices) - 1):
        x1 = int(hand_landmarks.landmark[middle_indices[i]].x * w)
        y1 = int(hand_landmarks.landmark[middle_indices[i]].y * h)
        x2 = int(hand_landmarks.landmark[middle_indices[i + 1]].x * w)
        y2 = int(hand_landmarks.landmark[middle_indices[i + 1]].y * h)
        cv2.line(image, (x1, y1), (x2, y2), color, 2)

# Draw a line connecting two measured points
def draw_connection_line(image, point1, point2, color=(0, 255, 255), thickness=2):
    cv2.line(image, point1, point2, color, thickness)

# Function to check if the hands are positioned correctly
def check_distance(distance,start_time):
    if distance < DISTANCE_CHECK:
        if start_time is None:
            start_time = time.time()  
        elapsed_time = time.time() - start_time
    else:
        start_time = None  
        elapsed_time = 0
    return elapsed_time, start_time

# Function to process the exercise Sit and Reach
def process_exercise(repeats):
    distances = []
    elapsed_time = None
    start_time = None
    last_detected_time = time.time() 

    # Main Loop
    while True:
        if kinect.has_new_color_frame():
            image, results, frame = process_frame(kinect)

            if results.left_hand_landmarks and results.right_hand_landmarks:
                last_detected_time = time.time()

                draw_landmarks(image, results)

                hand_landmark1 = results.left_hand_landmarks.landmark[12]  
                hand_landmark2 = results.right_hand_landmarks.landmark[12]

                right_hand = int(hand_landmark1.x * 640), int(hand_landmark1.y * 480)
                left_hand = int((hand_landmark2.x * 640)), int(hand_landmark2.y * 480)

                distance_pixel = calculate_distance_2d(right_hand, left_hand)
                distance = (distance_pixel * PIXEL_TO_CM_RATIO) - ERROR
                
                draw_connection_line(image, right_hand, left_hand)

                # distances.append(distance)
                # if len(distances) > AVERAGE_OVER:
                #     distances.pop(0)
                #     distance = average_distance(distances)

                elapsed_time,start_time = check_distance(distance,start_time) 

                cv2.putText(image, f"Dist: {distance:.2f} cm", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(image, f'Pos Right Hand: {right_hand[0]}, {right_hand[1]}', (1000, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)
                cv2.putText(image, f'Pos Left Hand: {left_hand[0]}, {left_hand[1]}', (1000, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

                if elapsed_time >= POSE_HELD_DURATION:

                    # if repeats in [0,1]:
                    #     if left_hand[1] <= right_hand[1]:
                    #         distance = -distance
                    # else:
                    #     if left_hand[1] <= right_hand[1]:
                    #         distance = -distance
                        
                    distance = -distance
                    
                    return f'{distance:.2f}'
            
            else:
                 if time.time() - last_detected_time >= POSE_NO_HELD_DURATION:
                    start_time = None

            cv2.imshow('Left Hand Tracking with Kinect and Holistic', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                finish_program()

# Function to show the register screen
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
                    active_field = i
                    break

    cv2.namedWindow("Cadastro")
    cv2.setMouseCallback("Cadastro", mouse_callback)

    while True:
        img = 255 * np.ones((400, 600, 3), dtype=np.uint8)

        for i, (x1, y1, x2, y2) in enumerate(positions):
            background_color = (230, 230, 230)
            cv2.rectangle(img, (x1, y1), (x2, y2), background_color, -1)
            border_color = (0, 255, 0) if i == active_field else (0, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2)
            cv2.putText(img, f"{fields[i]}:", (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(img, values[i], (x1 + 10, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        cv2.putText(img, "Aperte Enter para finalizar", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)

        cv2.imshow("Cadastro", img)
        key = cv2.waitKey(10) & 0xFF

        if key == 27:  
            finish_program()
        elif key == 13 or key == 10:  
            cv2.destroyAllWindows()
            return values
        elif key == 9:  # Tecla Tab
            active_field = (active_field + 1) % len(fields)
        elif active_field != -1:
            if key == 8:  
                values[active_field] = values[active_field][:-1]
            elif 32 <= key <= 126:  
                values[active_field] += chr(key)

# Function to show the real distance input screen
def real_distance():
    distancia = ""
    windown_width, windown_heigth = 600, 200

    cv2.namedWindow("Real Distance")

    while True:
        img = np.ones((windown_heigth, windown_width, 3), dtype=np.uint8) * 255

        cv2.rectangle(img, (50, 60), (550, 120), (230, 230, 230), -1)
        cv2.rectangle(img, (50, 60), (550, 120), (0, 0, 0), 2)

        cv2.putText(img, "Digite a Distância medida (cm):", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, distancia, (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.putText(img, "Pressione Enter para confirmar", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        cv2.imshow("Real Distance", img)
        key = cv2.waitKey(10) & 0xFF

        if key == 27: 
            cv2.destroyAllWindows()
            finish_program()
        elif key == 13 or key == 10:  
            if distancia:
                cv2.destroyAllWindows()
                return float(distancia.replace(",", ".")) 
        elif key == 8:  
            distancia = distancia[:-1]
        elif (key >= 48 and key <= 57) or key in [44, 46, 43, 45]:  
            distancia += chr(key)

repeats = 0

distances_right = []
distances_left = []

age,height,weight,gender = register()

gender = "Feminine" if gender == "F" else "Male"

while repeats < 4:
    final_distance = process_exercise(repeats)

    if final_distance is not None:
        
        real = real_distance()
        
        erro = np.abs(np.abs(float(real)) - np.abs(float(final_distance)))

        caminho_arquivo = "./tabelas_utentes/back_scratch_utentes.xlsx"
        df = pd.read_excel(caminho_arquivo, engine="openpyxl")

        new_line = {
            "Age": age,
            "Height": height,
            "Weight": weight,
            "Gender": gender,
            "Real distance": real,
            "Calculated distance": final_distance,
            "Erro": erro
        }

        df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
        df.to_excel(caminho_arquivo, index=False, engine="openpyxl")

        if repeats in [0,1]: 
            distances_right.append(final_distance)
            side = "right"
        else: 
            distances_left.append(final_distance)
            side = "left"

        with open("./logs_utentes/logs_back_scratch_utentes","a") as arquivo:
            arquivo.write(f"{dt.datetime.now()}, {age}, {height}, {weight}, {gender}, {real}, {final_distance},{side}\n")

        repeats += 1

        final_repetition_visualization(final_distance,real)
    else:
        print("Exercise not performed correctly")
        finish_program()

best_left,best_right = max(distances_left, key=float), max(distances_right, key=float)
final_visualization(best_left,best_right)

finish_program()