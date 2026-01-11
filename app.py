from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import glob
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
model = tf.keras.models.load_model("model/asl_model.h5")

# Global MediaPipe Hands instance (lazy loaded on first use)
_hands_detector = None


def get_hands_detector():
    """Get or create the global Hands detector instance"""
    global _hands_detector
    if _hands_detector is None:
        print("Initializing MediaPipe Hands detector...")
        _hands_detector = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
        )
        print("MediaPipe Hands detector ready")
    return _hands_detector


# Memory buffer to store frames (max 30 frames)
frame_buffer = deque(maxlen=30)  # JPG bytes


# Only keep sign_active for state
sign_active = False  # Track if hand signing capture is active

# Global variables for hand-tracking/transcribing
curr_word = ""
last_x = 0
total_x = 0
curr_x = 0
double_letter = False
curr_letter = ""
last_letter = ""
LETTERS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


# Remove camera detection logic


last_x = 0
total_x = 0
curr_x = 0
double_letter = False
curr_letter = "a"  # placeholder value --> will store actual current letter
last_letter = "b"  # ditto

# Callback function for sending ASL transcript (set by web.py to avoid circular import)
asl_transcript_callback = None


def set_asl_transcript_callback(callback):
    """Set the callback function for sending ASL transcripts"""
    global asl_transcript_callback
    asl_transcript_callback = callback


def create_default_image():
    """Create a default image when no camera is detected"""
    # Create a 1280x720 image with a dark gray background
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # Dark gray background

    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = "No Camera Detected"
    text2 = "Please connect a camera"

    # Calculate text size and position for centering
    text_size1 = cv2.getTextSize(text1, font, 2, 3)[0]
    text_size2 = cv2.getTextSize(text2, font, 1, 2)[0]

    text_x1 = (1280 - text_size1[0]) // 2
    text_y1 = (720 - text_size1[1]) // 2 - 50
    text_x2 = (1280 - text_size2[0]) // 2
    text_y2 = (720 + text_size2[1]) // 2 + 50

    # Draw text with white color
    cv2.putText(
        img, text1, (text_x1, text_y1), font, 2, (255, 255, 255), 3, cv2.LINE_AA
    )
    cv2.putText(
        img, text2, (text_x2, text_y2), font, 1, (200, 200, 200), 2, cv2.LINE_AA
    )

    return img


# Generate default image once at startup
default_image = create_default_image()


# Remove camera capture logic


# Remove frame generation logic (handled by host stream now)


# Remove camera thread and set_camera_id logic


def set_sign_active(active):
    """Set whether hand sign detection is active"""
    global sign_active
    sign_active = active


def transcribe(letter, double_letter, send):
    # this will transcribe letters
    global curr_word
    if letter == " ":
        return
    else:
        curr_word = curr_word + letter
        if double_letter == True:
            curr_word = curr_word + letter
        if send == True:
            if asl_transcript_callback is not None:
                asl_transcript_callback(curr_word)


def process_hand_data(hand):
    data_list = []
    # normalizes the data via the wrist for better model processing (wrist is (0,0,0))
    for measurement in hand.landmark:
        data_list.append(measurement.x - hand.landmark[0].x)
        data_list.append(measurement.y - hand.landmark[0].y)
        data_list.append(measurement.z - hand.landmark[0].z)
    return np.array(data_list).reshape(1, -1)


def get_letter(data):
    global curr_letter
    prediction = model.predict(data, verbose=0)
    id = np.argmax(prediction)
    curr_letter = LETTERS[id]


def capture_hands(curr_image):
    global last_letter
    global curr_letter
    global last_x
    global total_x
    global double_letter
    global send
    # this will be thing that stores all images, placeholder for now
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        # Read an image, flip it around y-axis for correct handedness output
        image = cv2.flip(cv2.imread(curr_image), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return
        primary_hand = results.multi_hand_landmarks[0]
        data = process_hand_data(primary_hand)
        get_letter(data)
        send = False
        if curr_letter != last_letter:
            if total_x >= 0.15:
                double_letter = True
            if curr_letter.isspace() == True:
                send = True
            if last_letter.isspace() == False:
                curr_letter = curr_letter.lower()
            transcribe(curr_letter, double_letter, send)
            last_x = 0
            total_x = 0
            double_letter = False
            last_letter = curr_letter
        # checks for double letters via wrist data
        if curr_letter == last_letter:
            curr_x = primary_hand.landmark[0].x
            total_x = total_x + (curr_x - last_x)
            last_x = curr_x
        curr_x = 0
