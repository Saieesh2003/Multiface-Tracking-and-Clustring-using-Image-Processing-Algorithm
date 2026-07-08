import cv2
import face_recognition
import numpy as np
import os



from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
# Sample data (replace with your actual data)
data = [[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]

# Create a DBSCAN object
dbscan = DBSCAN(eps=3, min_samples=2) 

# Fit the model to the data
labels = dbscan.fit_predict(data)

print(labels) 
# Initialize data storage
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
ENCODINGS_FILE = "face_encodings.npy"
ID_FILE = "face_ids.npy"
# Load known faces and IDs
if os.path.exists(ENCODINGS_FILE) and os.path.exists(ID_FILE):
    known_encodings = list(np.load(ENCODINGS_FILE, allow_pickle=True))
    known_ids = list(np.load(ID_FILE, allow_pickle=True))
else:
    known_encodings = []
    known_ids = []
# Generate new unique ID
def generate_unique_id(existing_ids):
    return max(existing_ids, default=0) + 1
# Initialize webcam
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame from the webcam
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        face_id = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            face_id = known_ids[match_index]
        else:
            new_id = generate_unique_id(known_ids)
            known_encodings.append(face_encoding)
            known_ids.append(new_id)

            np.save(ENCODINGS_FILE, np.array(known_encodings, dtype=object))
            np.save(ID_FILE, np.array(known_ids, dtype=object))
            face_id = new_id

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {face_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break

video_capture.release()
cv2.destroyAllWindows()