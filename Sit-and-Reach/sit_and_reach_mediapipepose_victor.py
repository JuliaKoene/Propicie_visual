import cv2
import mediapipe as mp
import numpy as np
import time
from pykinect2 import PyKinectRuntime, PyKinectV2
import math

# Initialize Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def adjust_fingertip_position(index):
    return np.array([index.x + 0, index.y - 0.02 ])  # Move 2 cm down (y-axis)

def adjust_figertoe_position(foot): 
    return np.array([foot.x + 0, foot.y + 0])

def calculate_distance_2d(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
def calcular_angulo(a, b, c):
    # Certifique-se de que 'a', 'b' e 'c' são objetos numpy.ndarray ou pontos do MediaPipe Landmark
    # Então, acesse as coordenadas x e y diretamente
    vetor_1 = (a[0] - b[0], a[1] - b[1])  # Vetor do ponto b para o ponto a
    vetor_2 = (c[0] - b[0], c[1] - b[1])  # Vetor do ponto b para o ponto c

    # Produto escalar
    produto_escalar = vetor_1[0] * vetor_2[0] + vetor_1[1] * vetor_2[1]

    # Normas (módulos)
    norma_vetor_1 = math.sqrt(vetor_1[0]**2 + vetor_1[1]**2)
    norma_vetor_2 = math.sqrt(vetor_2[0]**2 + vetor_2[1]**2)

    # Cálculo do cosseno do ângulo
    cos_angulo = produto_escalar / (norma_vetor_1 * norma_vetor_2)

    # Garantir que o valor de cos_angulo esteja no intervalo [-1, 1] para evitar erros de precisão
    cos_angulo = max(-1, min(1, cos_angulo))

    # Ângulo em radianos
    angulo_radianos = math.acos(cos_angulo)

    # Converter para graus
    angulo_graus = math.degrees(angulo_radianos)

    return angulo_graus

def average_distance(distances):
    return sum(distances) / len(distances)

def draw_dynamic_angle_arc(image, p1, p2, p3, angle):
    # Draw lines between points
    cv2.line(image, p1, p2, (255, 255, 0), 2)
    cv2.line(image, p2, p3, (255, 255, 0), 2)
    
    # Calculate the arc points
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Determine the arc center point
    arc_center = tuple(p2.astype(int))
    
    # Calculate radius and draw arc dynamically based on angle
    radius = int(np.linalg.norm(p1 - p2) / 2)
    
    # Draw filled arc
    axes = (radius, radius)
    cv2.ellipse(image, arc_center, axes, 0, 0, angle, (0, 255, 0), -1)
    cv2.ellipse(image, arc_center, axes, 0, 0, angle, (0, 0, 255), 2)


distances = []
average_over = 5
final_distance = None
pose_correct_start_time = None
pose_held_duration = 8  # seconds, increased for more time before the exercise ends

while True: 
    # numero = int(input("Digite um número inteiro: ")) 
    numero = 1       
    while True:
        if kinect.has_new_color_frame():  
            frame = kinect.get_last_color_frame()
            frame = frame.reshape((1080, 1920, 4))  # Kinect BGRA frame dimensions
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Correct BGRA to BGR conversion  py
            
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process the frame with MediaPipe Pose
            results = pose.process(rgb_frame)
            
            # Recolor back to BGR for rendering
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            distance = -1
            rigth_knee_angle = -1
            right_hip_angle = -1
            right_elbow_angle = -1
            pose_correct = "Incorrect"
            progress = 0

            if numero == 1:

                if results.pose_landmarks:

                    landmarks = results.pose_landmarks.landmark

                    # Check if all required landmarks are detected
                    required_landmarks = [
                        mp_pose.PoseLandmark.RIGHT_WRIST,
                        mp_pose.PoseLandmark.RIGHT_INDEX,
                        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
                        mp_pose.PoseLandmark.RIGHT_HIP,
                        mp_pose.PoseLandmark.RIGHT_KNEE,
                        mp_pose.PoseLandmark.RIGHT_ANKLE,
                        mp_pose.PoseLandmark.RIGHT_SHOULDER,
                        mp_pose.PoseLandmark.RIGHT_ELBOW
                    ]

                    if all(landmarks[lm.value].visibility > 0.0 for lm in required_landmarks):
                    
                        # Extract the necessary landmarks for distance calculation
                        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                        right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                        right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

                        # Adjust the right index position to simulate fingertip
                        right_fingertip = adjust_fingertip_position(right_index)

                        # Calculate the distance between right fingertip and right foot (2D)
                        right_foot_position = adjust_figertoe_position(right_foot)
                        distance = calculate_distance_2d(right_fingertip, right_foot_position)

                        for landmark in [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]:
                            cv2.circle(frame, (int(landmarks[landmark.value].x * frame.shape[1]), int(landmarks[landmark.value].y * frame.shape[0])), 10, (0, 255, 255), -1)
                        
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        distances.append(distance)
                        if len(distances) > average_over:
                            distances.pop(0)
                        distance = average_distance(distances)

                        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
                        right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
                        right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
                        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
                        right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
                        right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])

                        rigth_knee_angle = calcular_angulo(right_hip,right_knee,right_ankle)
                        right_hip_angle = calcular_angulo(right_shoulder,right_hip,right_knee)
                        right_elbow_angle = calcular_angulo(right_shoulder,right_elbow,right_wrist)

                        if 140 < right_elbow_angle < 180 and 10 < right_hip_angle < 140 and 152 < rigth_knee_angle < 177:
                            pose_correct = "Correct"
                            if pose_correct_start_time is None:
                                pose_correct_start_time = time.time()
                            progress = (time.time() - pose_correct_start_time) / pose_held_duration
                            if progress >= 1.0:
                                progress = 1.0
                                final_distance = distance
                                break
                        else:
                            pose_correct_start_time = None
                            progress = 0.0

                        shoulder_coords = tuple(np.multiply(right_shoulder[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                        knee_coords = tuple(np.multiply(right_knee[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                        hip_coords = tuple(np.multiply(right_hip[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                        ankle_coords = tuple(np.multiply(right_ankle[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                        elbow_coords = tuple(np.multiply(right_elbow[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                        wrist_coords = tuple(np.multiply(right_wrist[:2], [frame.shape[1], frame.shape[0]]).astype(int))

                        # draw_dynamic_angle_arc(frame,hip_coords, knee_coords, ankle_coords, rigth_knee_angle)
                        # cv2.putText(frame, f'Knee Angle: {rigth_knee_angle:.2f}', knee_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                        # draw_dynamic_angle_arc(frame, shoulder_coords, hip_coords, knee_coords, right_hip_angle)
                        # cv2.putText(frame, f'Hip Angle: {right_hip_angle:.2f}', hip_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # draw_dynamic_angle_arc(frame, shoulder_coords, elbow_coords, wrist_coords, elbow_angle)
                        # cv2.putText(frame, f'Elbow Angle: {elbow_angle:.2f}', elbow_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        cv2.putText(frame, f'Distance: {distance:.3f} m', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame,f'Foot Position: X: {right_foot.x:.3f} , Y:{right_foot.y:.3f} , Z:{right_foot.z}',(20,210),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)
                        cv2.putText(frame,f'Hand Postion: X: {right_index.x:.3f} , Y: {right_index.y:.3f}',(20,310),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        cv2.putText(frame, f'Pose: {pose_correct}', (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif numero == 2:

                if results.pose_landmarks:  

                    landmarks = results.pose_landmarks.landmark
                    
                    # Check if all required landmarks are detected
                    required_landmarks = [
                        mp_pose.PoseLandmark.LEFT_WRIST,
                        mp_pose.PoseLandmark.LEFT_INDEX,
                        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                        mp_pose.PoseLandmark.LEFT_HIP,
                        mp_pose.PoseLandmark.LEFT_KNEE,
                        mp_pose.PoseLandmark.LEFT_ANKLE,
                        mp_pose.PoseLandmark.LEFT_SHOULDER,
                        mp_pose.PoseLandmark.LEFT_ELBOW
                    ]

                    if all(landmarks[lm.value].visibility > 0.0 for lm in required_landmarks):
                    
                        # Extract the necessary landmarks for distance calculation
                        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                        left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
                        left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]

                        # Adjust the right index position to simulate fingertip
                        left_fingertip = adjust_fingertip_position(left_index)

                        # Calculate the distance between right fingertip and right foot (2D)
                        left_foot_position = np.array([left_foot.x, left_foot.y])
                        distance = calculate_distance_2d(left_fingertip, left_foot_position)

                        for landmark in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.LEFT_FOOT_INDEX]:
                            cv2.circle(frame, (int(landmarks[landmark.value].x * frame.shape[1]), int(landmarks[landmark.value].y * frame.shape[0])), 10, (0, 255, 255), -1)
                        
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        cv2.putText(frame, f'Distance: {distance:.3f} m', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame,f'Foot Position: X: {left_foot.x:.3f} , Y:{left_foot.y:.3f}',(20,210),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)
                        cv2.putText(frame,f'Hand Postion: X: {left_index.x:.3f} , Y: {left_index.y:.3f}',(20,310),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

            # Display the frame
            cv2.imshow('Chair Sit and Reach Exercise', frame)

            # Close the first window with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q') :
                exit()
            # Final result visualization
        
    if final_distance is not None:
            final_frame = np.zeros((500, 800, 3), dtype=np.uint8)  # Black screen for final results
            
            cv2.putText(final_frame, f'Exercise Completed', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(final_frame, f'Final Distance: {final_distance:.2f} meters', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(final_frame,f'Press "r" to restart or "q" to finish the exercise',(50,400),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,0),2)
            
            cv2.imshow('Final Results', final_frame)
            
            # Allow the user to close the final result window with 'q'
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):  # Pressione 'r' para reiniciar o exercício
                    cv2.destroyWindow('Final Results')
                    break
                elif key == ord('q'):  # Pressione 'q' para sair
                    exit()

    else:
        print("Exercise not performed correctly")
        break
    

kinect.close()
cv2.destroyAllWindows()