import cv2
import time
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import csv
from app import db  # Make sure to import your actual Flask app
from app import Student  # Import your Student model


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

# Load the face recognition model and cascade classifier
face_model = load_model("face_recognition_model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the holistic model from Mediapipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Initialize the video stream
video_stream = cv2.VideoCapture(0)
start_time = time.time()  # Record the start time

# Initialize attention_details as a dictionary of dictionaries
labels_dict = {0: "1032200746", 1: "1032200703", 2: "1032200614"}

# Create a dictionary to store attention details for each student
attention_details = {student_id: {"Open Eyes Time": 0, "Closed Eyes Time": 0, "Distracted Time": 0} for student_id in labels_dict.values()}

# Create the CSV file and write the header row
with open('attention_details.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Name", "Average Attention Span", "Total Time in seconds"])

current_student_id = None  # To track the current student
student_data_written = set()  # Set to keep track of students for whom data has been written

while True:
    frame_start_time = time.time()  # Start the frame timer

    ret, frame = video_stream.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Face detection using the cascade classifier
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process the frame with Mediapipe holistic model
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    frame_end_time = time.time()

    for (x, y, w, h) in faces:
        # Extract the face region
        face_region = gray_frame[y:y + h, x:x + w]

        # Resize the face region to match the model input size
        face_region = cv2.resize(face_region, (100, 100))
        face_region = np.reshape(face_region, (1, 100, 100, 1)) / 255.0

        # Predict the label using the face recognition model
        detected_student_label = np.argmax(face_model.predict(face_region), axis=1)[0]
        student_id = labels_dict.get(detected_student_label, "Unknown")

        if current_student_id != student_id:
            # If the student ID has changed and data hasn't been written, write the previous student's data
            if current_student_id is not None and current_student_id not in student_data_written:
                student_attention = attention_details[current_student_id]
                open_eyes_time = student_attention.get("Open Eyes Time", 0)
                closed_eyes_time = student_attention.get("Closed Eyes Time", 0)
                distracted_time = student_attention.get("Distracted Time", 0)
                if open_eyes_time + closed_eyes_time + distracted_time != 0:
                    attention_span = (open_eyes_time - distracted_time) / (
                            open_eyes_time + closed_eyes_time + distracted_time) * 100
                    with open('attention_details.csv', 'a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow([current_student_id, attention_span, total_time])
                        student_data_written.add(current_student_id)

            # Reset attention details for the new student
            current_student_id = student_id
            attention_details[current_student_id] = {"Open Eyes Time": 0, "Closed Eyes Time": 0,
                                                     "Distracted Time": 0}

        # Update the attention details for the corresponding student
        student_attention = attention_details[current_student_id]

        left_eye_lm = np.array([(point.x, point.y) for point in results.face_landmarks.landmark[33:41]])
        right_eye_lm = np.array([(point.x, point.y) for point in results.face_landmarks.landmark[263:271]])

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

        # Calculate the total time the camera was on
        total_time = time.time() - start_time

        if open_eyes_time + closed_eyes_time + distracted_time == 0:
            attention_span = 0
        else:
            attention_span = (open_eyes_time - distracted_time) / (
                    open_eyes_time + closed_eyes_time + distracted_time) * 100
        cv2.putText(image, f"Student ID: {current_student_id}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Attention: {attention_span:.2f}%", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Write the last student's data when exiting
if current_student_id is not None and current_student_id not in student_data_written:
    student_attention = attention_details[current_student_id]
    open_eyes_time = student_attention.get("Open Eyes Time", 0)
    closed_eyes_time = student_attention.get("Closed Eyes Time", 0)
    distracted_time = student_attention.get("Distracted Time", 0)
    if open_eyes_time + closed_eyes_time + distracted_time != 0:
        attention_span = (open_eyes_time - distracted_time) / (open_eyes_time + closed_eyes_time + distracted_time) * 100
        with open('attention_details.csv', 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([current_student_id, attention_span, total_time])

# Release resources and close the CSV file
cv2.destroyAllWindows()
video_stream.release()
