import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

# Path to the folder containing face images
IMAGE_PATH = "faces"

# Load known faces and names
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    try:
        for filename in os.listdir(IMAGE_PATH):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                print(f"Processing file: {filename}")  # Debugging line
                image = face_recognition.load_image_file(os.path.join(IMAGE_PATH, filename))
                encoding = face_recognition.face_encodings(image)
                if encoding:  # Check if encoding is not empty
                    known_face_encodings.append(encoding[0])
                    known_face_names.append(os.path.splitext(filename)[0])
                    print(f"Loaded face: {os.path.splitext(filename)[0]}")  # Debugging line
                else:
                    print(f"No encoding found for {filename}")  # Debugging line
    except Exception as e:
        print(f"Error loading faces: {e}")
    
    return known_face_encodings, known_face_names

# Initialize known faces
detected_faces = set()
known_face_encodings, known_face_names = load_known_faces()

# Open webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Open or create attendance file
attendance_file = "attendance.csv"
file_exists = os.path.exists(attendance_file)

with open(attendance_file, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Name", "Time"])  # Write header if file is new

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video frame")  # Debugging line
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        print(f"Detected {len(face_locations)} faces")  # Debugging line
        
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                name = known_face_names[match_index]
                
                if name not in detected_faces:
                    detected_faces.add(name)
                    writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    print(f"Logged attendance for: {name}")  # Debugging line
            
            # Draw rectangle around the face and label it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Face Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video_capture.release()
cv2.destroyAllWindows()
