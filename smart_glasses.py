import time
import os
import cv2
import numpy as np
import pytesseract
from gtts import gTTS
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from threading import Thread

# === GPIO PINS ===
TOUCH_PIN = 17
ULTRASONIC_TRIG = 23
ULTRASONIC_ECHO = 24

# === TESSERACT CONFIG ===
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# === SETUP GPIO ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(TOUCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(ULTRASONIC_TRIG, GPIO.OUT)
GPIO.setup(ULTRASONIC_ECHO, GPIO.IN)

# === CAMERA SETUP ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1280, 720)})
picam2.configure(config)
picam2.start()
time.sleep(2)

# === FACE RECOGNIZER ===
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists("trainer.yml"):
    recognizer.read("trainer.yml")

names = ["Unknown", "Person1", "Person2"] 

# === DISTANCE CHECK ===
def distance_check():
    while True:
        GPIO.output(ULTRASONIC_TRIG, True)
        time.sleep(0.00001)
        GPIO.output(ULTRASONIC_TRIG, False)

        start_time = time.time()
        stop_time = time.time()

        while GPIO.input(ULTRASONIC_ECHO) == 0:
            start_time = time.time()
        while GPIO.input(ULTRASONIC_ECHO) == 1:
            stop_time = time.time()

        time_elapsed = stop_time - start_time
        distance = (time_elapsed * 34300) / 2

        if distance < 30:
            print("[ALERT] Object too close!")

        time.sleep(1)

# === OCR + TTS ===
def capture_image_for_text():
    image = picam2.capture_array()
    rgb = image[:, :, 1:4][:, :, ::-1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh)
    return text

def speak_text(text):
    if text.strip():
        print("Speaking:", text)
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")
        os.system("mpg123 output.mp3")
    else:
        print("No text found.")

# === FACE RECOGNITION ===
def recognize_face():
    print("Face Recognition Mode")
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame[:, :, 1:4][:, :, ::-1], cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 80:
            print(f"Recognized: {names[id_]} (Confidence: {round(confidence, 2)})")
        else:
            print("Face detected: Unknown")

# === TOUCH HANDLER ===
def touch_listener():
    tap_count = 0
    last_tap_time = 0

    while True:
        if GPIO.input(TOUCH_PIN) == GPIO.LOW:
            now = time.time()
            if now - last_tap_time < 1:
                tap_count += 1
            else:
                tap_count = 1
            last_tap_time = now

            if tap_count == 2:
                print("Double tap detected")
                text = capture_image_for_text()
                speak_text(text)
                tap_count = 0
            elif tap_count == 1:
                print("Single tap detected")
                recognize_face()

            time.sleep(0.7)  # debounce delay

try:
    print("Starting Smart Glasses...")
    Thread(target=distance_check, daemon=True).start()
    touch_listener()
except KeyboardInterrupt:
    print("Exiting...")
finally:
    GPIO.cleanup()
    picam2.stop()