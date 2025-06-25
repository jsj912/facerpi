import cv2
import os 
import numpy as np
import pyttsx3
from picamera2 import Picamera2

# Text-to-Speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)   # Speed of speech
engine.setProperty('volume', 0.5)   # Max volume

# Parameters
font = cv2.FONT_HERSHEY_COMPLEX
height = 1
boxColor = (0, 0, 255)      # Red box
nameColor = (255, 255, 255) # White text
confColor = (255, 255, 0)   # Teal confidence

# Face detection and recognition setup
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Names corresponding to trained IDs
names = ['None', 'lola', 'Joan','ruhani']

# Camera setup
cam = Picamera2()
cam.preview_configuration.main.size = (640, 360)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.controls.FrameRate = 30
cam.preview_configuration.align()
cam.configure("preview")
cam.start()

last_spoken = None  # Track last person spoken

while True:
    # Capture a frame
    frame = cam.capture_array()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        frameGray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(150, 150)
    )

    for (x, y, w, h) in faces:
        namepos = (x + 5, y - 5)
        confpos = (x + 5, y + h - 5)
        cv2.rectangle(frame, (x, y), (x + w, y + h), boxColor, 3)

        # Recognize the face
        id, confidence = recognizer.predict(frameGray[y:y + h, x:x + w])

        if confidence < 30:
            id_name = names[id]
            confidence_text = f"{100 - confidence:.0f}%"

            if last_spoken != id_name:
                engine.say(f"Hello {id_name}")
                engine.runAndWait()
                last_spoken = id_name
        else:
            id_name = "Unknown"
            confidence_text = f"{100 - confidence:.0f}%"
            last_spoken = None

        # Display result
        cv2.putText(frame, str(id_name), namepos, font, height, nameColor, 2)
        cv2.putText(frame, str(confidence_text), confpos, font, height, confColor, 1)

    # Show the frame
    cv2.imshow('Raspi Face Recognizer', frame)

    # Key controls
    key = cv2.waitKey(100) & 0xff
    if key == 27 or key == 113:  # ESC or 'q'
        break

# Cleanup
print("\n [INFO] Exiting Program and cleaning up stuff")
cam.stop()
cv2.destroyAllWindows()
