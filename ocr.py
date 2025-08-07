import time
import os
import numpy as np
import pytesseract
from picamera2 import Picamera2
from gtts import gTTS
import cv2
import RPi.GPIO as GPIO

# === CONFIGURATION ===
BUTTON_PIN = 17  
DEBOUNCE_TIME = 0.3 
pytesseract.pytesseract.tesseract_cmd = ''  

# === SETUP GPIO ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def capture_image():
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1280, 720)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Warm-up time
        image = picam2.capture_array()
        picam2.stop()
        return image
    except Exception as e:
        print(f"[ERROR] Image capture failed: {e}")
        return None

def preprocess_image(rgb_image):
    try:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return rgb_image

def extract_text_from_image(image):
    try:
        rgb_image = image[:, :, 1:4][:, :, ::-1]  # Convert XRGB to RGB
        preprocessed = preprocess_image(rgb_image)
        text = pytesseract.image_to_string(preprocessed)
        return text
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        return ""

def speak_text(text):
    try:
        if text.strip():
            print("Detected text:\n", text)
            tts = gTTS(text=text, lang='en')
            tts.save("spoken_text.mp3")
            os.system("mpg123 spoken_text.mp3")
        else:
            print("No text detected.")
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")

def wait_for_button():
    print("Waiting for button press...")
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            print("Button pressed!")
            time.sleep(DEBOUNCE_TIME)
            return

def main():
    try:
        while True:
            wait_for_button()

            print("\nCapturing image...")
            image = capture_image()
            if image is None:
                continue

            print("Extracting text from image...")
            text = extract_text_from_image(image)

            print("Speaking text...")
            speak_text(text)

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
