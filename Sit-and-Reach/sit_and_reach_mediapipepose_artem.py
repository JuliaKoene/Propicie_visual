import cv2
import mediapipe as mp
import numpy as np
import time
from pykinect2 import PyKinectRuntime, PyKinectV2

# Initialize Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate Euclidean distance in 2D (x, y only)
def calculate_distance_2d(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to adjust the fingertip position a couple of centimeters forward
def adjust_fingertip_position(index, adjustment=0.02):
    return np.array([index.x, index.y - adjustment])  # Move 2 cm down (y-axis)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Function to calculate flexibility score with increased threshold and adjusted weights
def calculate_flexibility_score(distance, knee_angle, hip_angle, elbow_angle, max_distance=0.7):  # Increased distance threshold
    # Normalize the distance score (0 to 1)
    normalized_distance = (max_distance - distance) / max_distance
    normalized_distance = max(0.0, min(normalized_distance, 1.0))  # Clamp between 0 and 1
    
    # Normalize angles (closer to 180 is better for knee, hip, elbow)
    normalized_knee = (knee_angle - 150) / (180 - 150)
    normalized_hip = (hip_angle - 60) / (130 - 60)
    normalized_elbow = (elbow_angle - 160) / (180 - 160)
    
    # Adjust weights: focus more on distance, but slightly increase importance of angles
    flexibility_score = (normalized_distance * 0.75) + (normalized_knee * 0.1) + (normalized_hip * 0.1) + (normalized_elbow * 0.05)
    return max(0.0, min(flexibility_score, 1.0))

# Function to classify flexibility score as 'Bad', 'Good', or 'Nice'
def classify_flexibility_score(flexibility_score):
    if flexibility_score < 0.6:
        return "Bad"
    elif 0.6 <= flexibility_score < 0.85:
        return "Good"
    else:
        return "Nice"

# Function to draw dynamic angle arc on image
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

# Function to draw progress bar
def draw_progress_bar(image, progress, position, size=(200, 20)):
    cv2.rectangle(image, position, (position[0] + size[0], position[1] + size[1]), (255, 255, 255), 1)
    cv2.rectangle(image, position, (position[0] + int(size[0] * progress), position[1] + size[1]), (0, 255, 0), -1)

# Function to draw the flexibility bar based on combined distance and angles
def draw_flexibility_bar(image, flexibility_score, position=(50, 400), size=(300, 50)):
    # Define the weights for each section: 60% for Bad, 25% for Good, 15% for Nice
    bad_threshold = 0.6  # Bad section takes 60%
    good_threshold = 0.25  # Good section takes 25%
    nice_threshold = 0.15  # Nice section takes 15%

    # Draw the bar background
    cv2.rectangle(image, position, (position[0] + size[0], position[1] + size[1]), (255, 255, 255), 2)
    
    # Calculate the width of each section based on weights
    bad_width = int(size[0] * bad_threshold)
    good_width = int(size[0] * good_threshold)
    nice_width = size[0] - bad_width - good_width
    
    bad_section = (position[0], position[1], position[0] + bad_width, position[1] + size[1])
    good_section = (bad_section[2], position[1], bad_section[2] + good_width, position[1] + size[1])
    nice_section = (good_section[2], position[1], good_section[2] + nice_width, position[1] + size[1])
    
    # Fill the sections with different colors
    cv2.rectangle(image, (bad_section[0], bad_section[1]), (bad_section[2], bad_section[3]), (0, 0, 255), -1)  # Red for Bad
    cv2.rectangle(image, (good_section[0], good_section[1]), (good_section[2], good_section[3]), (0, 255, 255), -1)  # Yellow for Good
    cv2.rectangle(image, (nice_section[0], nice_section[1]), (nice_section[2], nice_section[3]), (0, 255, 0), -1)  # Green for Nice
    
    # Add text labels
    cv2.putText(image, 'Bad', (bad_section[0] + 10, bad_section[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, 'Good', (good_section[0] + 10, good_section[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, 'Nice', (nice_section[0] + 10, nice_section[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Calculate the line position within the bar based on flexibility score
    line_x = int(position[0] + flexibility_score * size[0])
    
    # Draw the line to represent the current flexibility
    cv2.line(image, (line_x, position[1]), (line_x, position[1] + size[1]), (255, 255, 255), 4)

# Function to calculate average distance
def average_distance(distances):
    return sum(distances) / len(distances)

# Variables to store distances for averaging
distances = []
average_over = 5
final_distance = None
pose_correct_start_time = None
pose_held_duration = 8  # seconds, increased for more time before the exercise ends

# Main loop to capture and display frames
while True:
    if kinect.has_new_color_frame():
        frame = kinect.get_last_color_frame()
        frame = frame.reshape((1080, 1920, 4))  # Kinect BGRA frame dimensions
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Correct BGRA to BGR conversion
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process the frame with MediaPipe Pose
        results = pose.process(rgb_frame)
        
        # Recolor back to BGR for rendering
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Initialize variables
        distance = -1
        knee_angle = -1
        hip_angle = -1
        elbow_angle = -1
        pose_correct = "Incorrect"
        progress = 0
        flexibility_score = 0  # Initialize flexibility_score to avoid undefined reference

        # Check if pose landmarks are detected
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
            
            if all(landmarks[lm.value].visibility > 0.5 for lm in required_landmarks):
                # Extract the necessary landmarks for distance calculation
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

                # Adjust the right index position to simulate fingertip
                right_fingertip = adjust_fingertip_position(right_index)
                
                # Calculate the distance between right fingertip and right foot (2D)
                right_foot_position = np.array([right_foot.x, right_foot.y])
                distance = calculate_distance_2d(right_fingertip, right_foot_position)
                
                # Append distance to list and calculate average
                distances.append(distance)
                if len(distances) > average_over:
                    distances.pop(0)
                distance = average_distance(distances)
                
                # Extract the necessary landmarks for angle calculation
                right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
                right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
                right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
                right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
                right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
                right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
                
                # Calculate the angles of the right knee, hip, and elbow
                knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                # Calculate flexibility score based on distance and angles
                flexibility_score = calculate_flexibility_score(distance, knee_angle, hip_angle, elbow_angle)
                
                # Revert pose correctness thresholds to original and add elbow angle check
                if 150 < knee_angle < 180 and 60 < hip_angle < 130 and 160 < elbow_angle < 180:  # Added elbow check
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
                
                # Render detections with highlighted joints
                for landmark in [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]:
                    cv2.circle(frame, (int(landmarks[landmark.value].x * frame.shape[1]), int(landmarks[landmark.value].y * frame.shape[0])), 10, (0, 255, 255), -1)  # Yellow color for feedback
                
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                
                # Visualize the distance
                cv2.putText(frame, f'Distance: {distance:.2f} m', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Visualize the knee angle with dynamic filled parabola
                knee_coords = tuple(np.multiply(right_knee[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                hip_coords = tuple(np.multiply(right_hip[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                ankle_coords = tuple(np.multiply(right_ankle[:2], [frame.shape[1], frame.shape[0]]).astype(int))
                draw_dynamic_angle_arc(frame, hip_coords, knee_coords, ankle_coords, knee_angle)
                cv2.putText(frame, f'Knee Angle: {knee_angle:.2f}', knee_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Visualize if the pose is correct
                cv2.putText(frame, f'Pose: {pose_correct}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the progress bar
        draw_progress_bar(frame, progress, position=(10, 200))
        
        # Display the flexibility bar based on the flexibility score
        draw_flexibility_bar(frame, flexibility_score, position=(50, 400), size=(300, 50))
        
        # Display the frame
        cv2.imshow('Chair Sit and Reach Exercise', frame)

    # Close the first window with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final result visualization
if final_distance is not None:
    final_frame = np.zeros((500, 800, 3), dtype=np.uint8)  # Black screen for final results
    flexibility_result = classify_flexibility_score(flexibility_score)
    
    cv2.putText(final_frame, f'Exercise Completed', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(final_frame, f'Final Distance: {final_distance:.2f} meters', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(final_frame, f'Flexibility Bar Result: {flexibility_score:.2f} ({flexibility_result})', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Final Results', final_frame)
    
    # Allow the user to close the final result window with 'q'
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

else:
    print("Exercise not performed correctly")

kinect.close()
cv2.destroyAllWindows()
