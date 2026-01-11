from collections import deque
import cv2
import mediapipe as mp
# mediapipe changed its top-level API in 0.10+: the legacy `mp.solutions` API
# may not be available. Try to use it if present; otherwise fall back to
# disabling landmark drawing so the server can run. The app can be extended
# later to use `mediapipe.tasks.python` for landmarking if desired.
try:
	mp_drawing = mp.solutions.drawing_utils
	mp_drawing_styles = mp.solutions.drawing_styles
	mp_hands = mp.solutions.hands
	mediapipe_has_solutions = True
except Exception:
	mp_drawing = None
	mp_drawing_styles = None
	mp_hands = None
	mediapipe_has_solutions = False
	print("Warning: mediapipe.solutions not available; hand landmark drawing disabled.")
import numpy as np
import threading
import time
import glob
import tensorflow as tf
from multiprocessing import Process, Queue

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 1. CREATE A CENTRALIZED THREAD-SAFE GESTURE PROCESSOR
from queue import Queue
from dataclasses import dataclass
from typing import Optional

@dataclass
class GestureFrame:
    timestamp: float
    hand_landmarks: list
    frame_id: int

class ThreadSafeGestureProcessor:
    def __init__(self, model_path):
        self.lock = threading.RLock()
        self.model = tf.keras.models.load_model(model_path)
        self.gesture_queue = Queue(maxsize=30)
        self.current_word = ""
        self.gesture_buffer = []
        self.last_gesture = None
        self.silence_threshold = 0.5  # seconds
        self.last_gesture_time = 0
        
    def add_gesture(self, frame: GestureFrame):
        """Thread-safe gesture addition"""
        try:
            self.gesture_queue.put_nowait(frame)
        except:
            pass  # Queue full, drop frame
    
    def process(self):
        """Run in dedicated processing thread"""
        while True:
            try:
                frame = self.gesture_queue.get(timeout=1.0)
                with self.lock:
                    self._process_landmarks(frame)
                    self._check_word_boundary()
            except:
                with self.lock:
                    self._check_word_boundary()  # Force boundary on silence
    
    def get_transcript(self):
        """Thread-safe transcript retrieval"""
        with self.lock:
            return self.current_word
    
    def reset(self):
        """Thread-safe reset"""
        with self.lock:
            self.current_word = ""
            self.gesture_buffer = []

# 2. SEPARATE FRAME CAPTURE FROM HAND PROCESSING
# - Frame capture thread: just reads camera frames into a thread-safe buffer
# - Hand processing thread: reads from buffer, detects hands, predicts letters
# - Gesture processing thread: accumulates letters into words, sends callbacks

# 3. USE PROPER SYNCHRONIZATION FOR GLOBAL STATE
# Replace global variables with a thread-safe state manager:
class AppState:
    def __init__(self):
        self.lock = threading.RLock()
        self._sign_active = True
        self._connected_clients = 0
        self._host_logged_in = False
    
    @property
    def sign_active(self):
        with self.lock:
            return self._sign_active
    
    @sign_active.setter
    def sign_active(self, value):
        with self.lock:
            self._sign_active = value
    
    # Similar for other properties...

# 4. IMPROVE WORD BOUNDARY DETECTION
# Instead of motion tracking (total_x), use:
# - Gesture duration: If same letter for >500ms, it's a double letter
# - Silence detection: If no hand detected for >500ms, end word
# - Space gesture: Define specific gesture as word separator (or use timeout)

# 5. ENHANCE MODEL INTEGRATION
# Load model in dedicated thread and use thread-safe inference:
class ThreadSafeModel:
    def __init__(self, model_path):
        self.lock = threading.Lock()
        self.model = tf.keras.models.load_model(model_path)
        self.graph = tf.compat.v1.get_default_graph()
    
    def predict(self, data):
        with self.lock:
            with self.graph.as_default():
                return self.model.predict(data, verbose=0)


# Global MediaPipe Hands instance (lazy loaded on first use)
_hands_detector = None
_hands_detector_lock = threading.Lock()

def get_hands_detector():
	"""Get or create the global Hands detector instance (thread-safe)"""
	global _hands_detector
	with _hands_detector_lock:
		if _hands_detector is None:
			print("Initializing MediaPipe Hands detector...")
			_hands_detector = mp_hands.Hands(
				static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
			)
			print("MediaPipe Hands detector ready")
		return _hands_detector


# Memory buffer to store frames (max 30 frames)
frame_buffer = deque(maxlen=30)  # JPG bytes
frame_byte_q = Queue(maxsize=5)


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


def detect_available_cameras():
	"""Detect available camera devices and return list of camera IDs"""
	available_cameras = []
	for video_device in sorted(glob.glob("/dev/video*")):
		try:
			device_num = int(video_device.split("video")[1])
			cap_test = cv2.VideoCapture(device_num, cv2.CAP_V4L2)
			time.sleep(0.1)
			if cap_test.isOpened():
				available_cameras.append(device_num)
				cap_test.release()
		except Exception:
			pass
	return available_cameras


def set_default_camera():
	"""Set camera_id to first available camera, or keep existing"""
	global camera_id
	available = detect_available_cameras()
	if available:
		camera_id = available[0]
		print(f"Found {len(available)} camera(s). Using camera {camera_id} (/dev/video{camera_id})")
		if len(available) > 1:
			print(f"Other cameras available: {[f'/dev/video{c}' for c in available[1:]]}")
	else:
		print("No cameras detected; will use default image until a camera is connected.")


def capture_frames():
	"""Continuously capture frames from USB webcam (Raspberry Pi compatible)"""
	global cap, current_frame, camera_available, camera_id, stop_capture, model_frame

	# Helper to try opening a device id
	def try_open(dev_id):
		c = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
		time.sleep(0.2)
		if c.isOpened():
			return c
		try:
			c.release()
		except Exception:
			pass
		return None

	# First try the configured camera_id, then probe available devices
	cap = try_open(camera_id)
	if cap is None:
		available = detect_available_cameras()
		for dev in available:
			cap = try_open(dev)
			if cap is not None:
				camera_id = dev
				break

	if cap is None or not cap.isOpened():
		print(f"WARNING: No USB camera detected on /dev/video{camera_id}. Using default image.")
		camera_available = False
		with frame_lock:
			current_frame = default_image.copy()
	else:
		camera_available = True
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
		cap.set(cv2.CAP_PROP_FPS, 15)
		try:
			cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		except Exception:
			pass
		print(f"USB camera detected and initialized successfully on /dev/video{camera_id}.")

	while True:
		if camera_available:
			ret, frame = cap.read()
			if not ret:
				print("Failed to read frame from camera. Switching to default image.")
				camera_available = False
				with frame_lock:
					current_frame = default_image.copy()
				break

			if stop_capture:
				print("Stopping camera capture...")
				break

			with frame_lock:
				current_frame = frame.copy()
				# Store frame in buffer (jpeg bytes)
				_, jpeg = cv2.imencode(".jpg", frame)
				frame_buffer.append(jpeg.tobytes())
				# Prepare a model-ready RGB float32 frame (224x224 normalized)
				try:
					rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					small = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
					model_frame = small.astype(np.float32) / 255.0
				except Exception:
					model_frame = None
				print(f"Got image - Buffer size: {len(frame_buffer)}")
		else:
			with frame_lock:
				current_frame = default_image.copy()
				model_frame = None
			time.sleep(0.1)

	if cap is not None:
		try:
			cap.release()
		except Exception:
			pass


def generate_frames():
	"""Generate frames for streaming and optionally process hands"""
	# Only create the legacy `Hands` detector if the `mp.solutions` API
	# is available. If not, skip hand processing (server still runs).
	hands = None
	if mediapipe_has_solutions and mp_hands is not None:
		hands = mp_hands.Hands(
			static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
		)

	while True:
		with frame_lock:
			# Use default image if current_frame is None or camera is not available
			frame_to_send = (
				current_frame if current_frame is not None else default_image
			)
			frame_to_process = frame_to_send.copy()
			is_default_image = not camera_available

		# Process hand detection if signing is active and the legacy
		# `mp.solutions` API is available.
		if (
			sign_active
			and frame_to_process is not None
			and not is_default_image
			and hands is not None
		):
			try:
				# Convert BGR to RGB for MediaPipe
				frame_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
				results = hands.process(frame_rgb)

				# Draw hand landmarks if detected and drawing utils are present
				if (
					results is not None
					and getattr(results, "multi_hand_landmarks", None)
					and mp_drawing is not None
				):
					for hand_landmarks in results.multi_hand_landmarks:
						try:
							mp_drawing.draw_landmarks(
								frame_to_process,
								hand_landmarks,
								mp_hands.HAND_CONNECTIONS,
								mp_drawing_styles.get_default_hand_landmarks_style(),
								mp_drawing_styles.get_default_hand_connections_style(),
							)
						except Exception:
							# drawing failures shouldn't crash the server
							pass
					# Update frame_to_send with drawn landmarks
					frame_to_send = frame_to_process
			except Exception as e:
				print(f"Error processing hand landmarks: {e}")

		_, jpeg = cv2.imencode(".jpg", frame_to_send)
		frame_bytes = jpeg.tobytes()

		yield (
			b"--frame\r\n"
			b"Content-Type: image/jpeg\r\n"
			b"Content-Length: "
			+ str(len(frame_bytes)).encode()
			+ b"\r\n\r\n"
			+ frame_bytes
			+ b"\r\n"
		)

		# Add delay - 1 second for default image, ~30fps for camera feed
		if is_default_image:
			time.sleep(0.1)  # 1 second delay for default image
		else:
			time.sleep(0.033)  # ~30 fps for camera feed
	hands.close()


def start_camera_capture():
	"""Start or restart camera capture thread"""
	global capture_thread, stop_capture

	# Ensure default camera selection is up-to-date
	set_default_camera()

	# Stop existing thread if running
	if capture_thread and capture_thread.is_alive():
		stop_capture = True
		capture_thread.join(timeout=2.0)
		stop_capture = False

	# Start new capture thread
	capture_thread = threading.Thread(target=capture_frames, daemon=True)
	capture_thread.start()


def set_camera_id(new_id):
	"""Set the camera ID"""
	global camera_id
	camera_id = new_id


def set_sign_active(active):
	# Set whether hand sign detection is active
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


def capture_hands(frame_byte_q):
	global last_letter
	global curr_letter
	global last_x
	global total_x
	global double_letter
	global send
	while True:
		curr_image = frame_byte_q.get(block=True)
		with get_hands_detector() as hands:
			# Read an image, flip it around y-axis for correct handedness output
			nparr = np.frombuffer(curr_image, np.uint8)
			image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			if image is None:
				return
			image = cv2.flip(image, 1)
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
