import cv2
import os
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the DNN model for face detection
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Create a directory to save images if it doesn't exist
save_path = "face_images"
if not os.path.exists(save_path):
    os.mkdir(save_path)

# ID for each individual
person_id = input("Enter the ID for the person you're collecting data for: ")

# Create a directory for this individual
person_path = os.path.join(save_path, person_id)
if not os.path.exists(person_path):
    os.mkdir(person_path)

# Initialize image counter
img_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Prepare the image for face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Detect faces
    net.setInput(blob)
    detections = net.forward()

    # Loop through detections and save face images
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            # Save the captured face
            face_crop = frame[startY:endY, startX:endX]
            face_filename = os.path.join(person_path, f"{img_count}.jpg")
            cv2.imwrite(face_filename, face_crop)
            img_count += 1

    # Display the resulting frame
    cv2.imshow("Image Collection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
