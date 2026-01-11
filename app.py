from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import glob
import tensorflow as tf
from multiprocessing import Process, Queue

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Memory buffer to store frames (max 30 frames)
frame_buffer = deque(maxlen=30)  # JPG bytes
global frame_byte_q
frame_byte_q = Queue(maxsize=1)


# Only keep sign_active for state
sign_active = True  # Track if hand signing capture is active (default ON)

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
	text1 = "No Stream Online"
	text2 = "Host can login to start"

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


def set_sign_active(active):
	# Set whether hand sign detection is active
	global sign_active
	sign_active = active


def transcribe(letter, double_letter, send):
	# this will transcribe letters
	global curr_word
	curr_word = curr_word + letter
	if asl_transcript_callback is not None:
		asl_transcript_callback(letter)
		print(f"Sent ASL Transcript: {letter}")
	else:
		print("ASL Transcript Callback not set!")
	"""if letter == " ":
		return
	else:
		curr_word = curr_word + letter
		if double_letter == True:
			curr_word = curr_word + letter
		if send == True:
			if asl_transcript_callback is not None:
				asl_transcript_callback(curr_word)
				print(f"Sent ASL Transcript: {curr_word}")
			else:
				print("ASL Transcript Callback not set!")
    """


def process_hand_data(hand):
	data_list = []
	# normalizes the data via the wrist for better model processing (wrist is (0,0,0))
	for measurement in hand.landmark:
		data_list.append(measurement.x - hand.landmark[0].x)
		data_list.append(measurement.y - hand.landmark[0].y)
		data_list.append(measurement.z - hand.landmark[0].z)
	return np.array(data_list).reshape(1, -1)


def get_letter(data, model):
	global curr_letter
	print("predicting letter")
	prediction = model.predict(data, verbose=1)
	print("predicted")
	id = np.argmax(prediction)

	return LETTERS[id]


def capture_hands(frame_byte_q):
    global last_letter
    global curr_letter
    global last_x
    global total_x
    global double_letter
    global send
    model = tf.keras.models.load_model("model/asl_model.h5")
    frame_counter = 0
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )
    print("Starting capture hands thread")
    last_letter = ""
    while True:
        curr_image = frame_byte_q.get(block=True)
        frame_counter += 1
        print(f"[capture_hands] Processing frame {frame_counter}")
        # Read an image, flip it around y-axis for correct handedness output
        nparr = np.frombuffer(curr_image, np.uint8)
        if nparr is None:
            print("NP Array failed")
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[capture_hands] Dropped frame {frame_counter}: image decode failed")
            continue
        image = cv2.flip(image, 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            print(
				f"[capture_hands] Dropped frame {frame_counter}: no hand landmarks detected"
			)
            curr_letter = ""
            continue
        primary_hand = results.multi_hand_landmarks[0]
        data = process_hand_data(primary_hand)
        curr_letter = get_letter(data, model)
        # send = False
        if curr_letter != last_letter:
            """if total_x >= 0.15:
                double_letter = True
            if curr_letter.isspace() == True:
                send = True
            """
            transcribe(curr_letter, double_letter, True)
            last_x = 0
            total_x = 0
            double_letter = False
        # checks for double letters via wrist data
        """if curr_letter == last_letter:
            curr_x = primary_hand.landmark[0].x
            total_x = total_x + (curr_x - last_x)
            last_x = curr_x
            curr_x = 0
		"""
        last_letter = curr_letter
        print(f"Processeed frame {frame_counter}!")
