import cv2
import os
from picamera2 import Picamera2

# Constants
COUNT_LIMIT = 300
POS = (30, 60)  # Text position
FONT = cv2.FONT_HERSHEY_COMPLEX
HEIGHT = 1.5
TEXTCOLOR = (0, 0, 255)  # Red
BOXCOLOR = (255, 0, 255)  # Pink
WEIGHT = 3
FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get face ID
face_id = input('\n----Enter User-id and press <return>---- ')
print("\n[INFO] Initializing face capture. Look at the camera and wait!")

# Create folders
os.makedirs("dataset", exist_ok=True)
os.makedirs("old_dataset", exist_ok=True)

# Initialize camera
cam = Picamera2()
cam.preview_configuration.main.size = (640, 360)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.controls.FrameRate = 30
cam.preview_configuration.align()
cam.configure("preview")

try:
    cam.start()
except Exception as e:
    print(f"[ERROR] Failed to start camera: {e}")
    exit(1)

count = 0

while True:
    frame = cam.capture_array()
    cv2.putText(frame, f'Count: {count}', POS, FONT, HEIGHT, TEXTCOLOR, WEIGHT)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOXCOLOR, 3)
            count += 1

            file_path = os.path.join("dataset", f"User.{face_id}.{count}.jpg")

            if os.path.exists(file_path):
                old_file_path = file_path.replace("dataset", "old_dataset")
                os.rename(file_path, old_file_path)

            cv2.imwrite(file_path, gray[y:y + h, x:x + w])
            print(f"[INFO] Saved image: {file_path}")

            if count >= COUNT_LIMIT:
                print(f"[INFO] Collected {COUNT_LIMIT} face samples.")
                break
    else:
        print("[INFO] No face detected, image not saved.")

    # Show frame (optional - works only if connected to monitor)
    cv2.imshow('FaceCapture', frame)

    key = cv2.waitKey(100) & 0xff
    if key == 27 or key == ord('q') or count >= COUNT_LIMIT:  # ESC or 'q'
        print("[INFO] Exiting capture loop.")
        break

# Cleanup
print("\n[INFO] Exiting Program and cleaning up...")
cam.stop()
cv2.destroyAllWindows()
