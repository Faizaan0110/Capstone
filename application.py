import sys
import cv2
import time
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import csv

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def gaze_direction(eye):
    eye_center = ((eye[0] + eye[3]) / 2)
    direction = eye[0] - eye_center
    return direction

def head_pose(landmarks):
    nose = np.array([(landmarks.landmark[4].x, landmarks.landmark[4].y)])
    left_eye_inner = np.array([(landmarks.landmark[33].x, landmarks.landmark[33].y)])
    right_eye_inner = np.array([(landmarks.landmark[263].x, landmarks.landmark[263].y)])
    return nose, left_eye_inner, right_eye_inner

face_model = load_model("face_recognition_model.h5")
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
video_stream = cv2.VideoCapture(0)
time.sleep(2.0)

# Initialize attention_details as a dictionary of dictionaries
labels_dict = {0: "1032200614", 1: "1032200703", 2: "1032200746"}

# Prompt to input the subject as a command line argument
if len(sys.argv) < 2:
    print("Please provide the subject as a command line argument.")
    sys.exit(1)

subject = sys.argv[1]

# Create a dictionary to store attention details for each student
attention_details = {student_id: {"Open Eyes Time": 0, "Closed Eyes Time": 0, "Distracted Time": 0, "Frame Time": 0.0, "Subject": subject} for student_id in labels_dict.values()}

# Create the CSV file and write the header row
with open('attention_details.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Name", "Average Attention Span", "Time in seconds", "Subject"])

current_student_id = None  # To track the current student

while True:
    frame_start_time = time.time()  # Start the frame timer

    ret, frame = video_stream.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    frame_end_time = time.time()
    elapsed_time = frame_end_time - frame_start_time

    if results.face_landmarks:
        left_eye_lm = np.array([(point.x, point.y) for point in results.face_landmarks.landmark[33:41]])
        right_eye_lm = np.array([(point.x, point.y) for point in results.face_landmarks.landmark[263:271]])

        # Detect the current student label based on your system's logic
        detected_student_label = 0  # Replace with your logic

        student_id = labels_dict[detected_student_label]

        if current_student_id != student_id:
            # If the student ID has changed, write the previous student's data
            if current_student_id is not None:
                student_attention = attention_details[current_student_id]
                open_eyes_time = student_attention.get("Open Eyes Time", 0)
                closed_eyes_time = student_attention.get("Closed Eyes Time", 0)
                distracted_time = student_attention.get("Distracted Time", 0)
                frame_time = student_attention.get("Frame Time", 0.0)
                if open_eyes_time + closed_eyes_time + distracted_time != 0:
                    attention_span = (open_eyes_time - distracted_time) / (open_eyes_time + closed_eyes_time + distracted_time) * 100
                    with open('attention_details.csv', 'a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow([current_student_id, attention_span, frame_time, student_attention["Subject"]])

            # Reset attention details for the new student
            current_student_id = student_id
            attention_details[current_student_id] = {"Open Eyes Time": 0, "Closed Eyes Time": 0, "Distracted Time": 0, "Frame Time": 0.0, "Subject": subject}

        # Update the attention details for the corresponding student
        student_attention = attention_details[current_student_id]

        left_ear = eye_aspect_ratio(left_eye_lm)
        right_ear = eye_aspect_ratio(right_eye_lm)
        average_ear = (left_ear + right_ear) / 2.0

        if average_ear > 0.3:
            student_attention["Open Eyes Time"] += 1
        else:
            student_attention["Closed Eyes Time"] += 1

        left_gaze = gaze_direction(left_eye_lm)
        right_gaze = gaze_direction(right_eye_lm)

        nose, left_eye_inner, right_eye_inner = head_pose(results.face_landmarks)
        pose_offset = np.linalg.norm(nose - (left_eye_inner + right_eye_inner) / 2)

        if np.linalg.norm(left_gaze) > 0.05 or np.linalg.norm(right_gaze) > 0.05 or pose_offset > 0.1:
            student_attention["Distracted Time"] += 1

        open_eyes_time = student_attention.get("Open Eyes Time", 0)
        closed_eyes_time = student_attention.get("Closed Eyes Time", 0)
        distracted_time = student_attention.get("Distracted Time", 0)
        student_attention["Frame Time"] += elapsed_time

        if open_eyes_time + closed_eyes_time + distracted_time == 0:
            attention_span = 0
        else:
            attention_span = (open_eyes_time - distracted_time) / (open_eyes_time + closed_eyes_time + distracted_time) * 100

        cv2.putText(image, f"Student ID: {current_student_id}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Attention: {attention_span:.2f}%", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Subject: {subject}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Write the last student's data when exiting
if current_student_id is not None:
    student_attention = attention_details[current_student_id]
    open_eyes_time = student_attention.get("Open Eyes Time", 0)
    closed_eyes_time = student_attention.get("Closed Eyes Time", 0)
    distracted_time = student_attention.get("Distracted Time", 0)
    frame_time = student_attention.get("Frame Time", 0.0)
    if open_eyes_time + closed_eyes_time + distracted_time != 0:
        attention_span = (open_eyes_time - distracted_time) / (open_eyes_time + closed_eyes_time + distracted_time) * 100
        with open('attention_details.csv', 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([current_student_id, attention_span, frame_time, student_attention["Subject"]])

# Release resources and close the CSV file
cv2.destroyAllWindows()
video_stream.release()

