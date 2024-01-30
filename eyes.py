import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2
import dlib 
from collections import deque

# Initialize a buffer with a maximum length
prediction_buffer = deque(maxlen=5)

def DrowsinessModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/Neha/Desktop/Drowsiness/shape_predictor_68_face_landmarks.dat")  # Replace with the actual path




def eye_aspect_ratio(eye):
    # Convert the eye landmarks to NumPy array
    eye = np.array(eye, dtype=float)

    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # Return the eye aspect ratio
    return ear


def show_webcam():
    face_cascade = cv2.CascadeClassifier('C:/Users/Neha/Desktop/drowsiness/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get drowsiness state for the frame
        drowsiness_state = detect_drowsiness(frame, gray, predictor)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Display rectangle around faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Display drowsiness state text above the rectangle
            cv2.putText(frame, drowsiness_state, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        cv2.imshow('Video', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_drowsiness(frame, gray, predictor):
    # Convert the frame to grayscale for face detection
    faces = detector(gray)

    # Loop over the detected faces
    for face in faces:
        # Get the facial landmarks
        shape = predictor(gray, face)

        # Extract eye coordinates from the facial landmarks
        left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
        right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]

        # Calculate eye aspect ratio (EAR) for left and right eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Use a threshold for EAR to determine drowsiness
        ear_threshold = 0.2  # Adjust as needed
        if left_ear < ear_threshold and right_ear < ear_threshold:
            prediction_buffer.append('Drowsy')
        else:
            prediction_buffer.append('Active')

    # Perform majority vote if there are votes
    if prediction_buffer:
        drowsy_votes = prediction_buffer.count('Drowsy')
        active_votes = prediction_buffer.count('Active')

        if drowsy_votes > active_votes:
            return 'Wake up you look Drowsy!'
        else:
            return 'Great you look Active!'

    # No face detected
    return 'No Face Detected'




top = tk.Tk()
top.geometry('800x600')
top.title('Drowsiness Detector')
top.configure(background='#CDCDCD')
label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('C:/Users/Neha/Desktop/drowsiness/haarcascade_frontalface_default.xml')
model = DrowsinessModel("model_a.json", "model_a_weights.h5")

def start_realtime_detection():
    show_webcam()

realtime_btn = Button(top, text="Start Real-Time Detection", command=start_realtime_detection, padx=10, pady=5)
realtime_btn.configure(background="#364156", foreground='white', font=('arial', 12, 'bold'))
realtime_btn.pack(side='bottom', pady=20)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Drowsiness Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()
