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

# (mp_drawing, mp_drawing_styles, mp_hands) already set above

# Memory buffer to store frames (max 30 frames)
frame_buffer = deque(maxlen=30)  # JPG bytes

# Global variables for video capture
cap = None
current_frame = None  # JPG
frame_lock = threading.Lock()
camera_available = False
camera_id = 0  # Default camera ID
capture_thread = None
stop_capture = False
sign_active = False  # Track if hand signing capture is active
# Latest model-ready frame (numpy array, RGB, float32 normalized to [0,1])
model_frame = None
asl_transcript_callback = None
inference_thread = None
inference_stop = False


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

	if hands is not None:
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
	"""Set whether hand sign detection is active"""
	global sign_active
	sign_active = active
	# Start inference thread when enabling sign capture
	if sign_active:
		start_sign_inference()


def get_model_input():
	"""Return the latest model-ready frame (RGB float32 224x224) or None."""
	global model_frame
	with frame_lock:
		if model_frame is None:
			return None
		return model_frame.copy()


def get_recent_frames(count: int = 30):
	"""Return up to `count` most recent frames from the buffer as model-ready arrays.

	Returns a list of numpy arrays shaped (224,224,3) dtype float32 normalized [0,1].
	"""
	results = []
	with frame_lock:
		items = list(frame_buffer)[-count:]
	for b in items:
		try:
			arr = np.frombuffer(b, dtype=np.uint8)
			img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
			if img is None:
				continue
			rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			small = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
			results.append(small.astype(np.float32) / 255.0)
		except Exception:
			continue
	return results


def get_sequence_for_model(seq_len: int = 30):
	"""Return a numpy array shaped (N,224,224,3) with the latest up to seq_len frames.

	If fewer than `seq_len` frames are available, returns the available frames.
	"""
	frames = get_recent_frames(seq_len)
	if not frames:
		return None
	return np.stack(frames, axis=0)


def set_asl_transcript_callback(callback):
	"""Set a callback that receives a sequence numpy array when inference runs.

	Callback signature: fn(sequence: np.ndarray) -> None
	"""
	global asl_transcript_callback
	asl_transcript_callback = callback


def sign_inference_loop(poll_interval: float = 0.5, seq_len: int = 30):
	"""Background loop that collects sequences while `sign_active` is True and
	invokes the `asl_transcript_callback` with the sequence for downstream ML.
	"""
	global inference_stop
	while not inference_stop:
		if not sign_active:
			time.sleep(poll_interval)
			continue

		seq = get_sequence_for_model(seq_len)
		if seq is not None:
			try:
				if asl_transcript_callback is not None:
					asl_transcript_callback(seq)
			except Exception as e:
				print(f"Error in ASL callback: {e}")

		time.sleep(poll_interval)


def start_sign_inference():
	"""Start the inference background thread if not already running."""
	global inference_thread, inference_stop
	if inference_thread and inference_thread.is_alive():
		return
	inference_stop = False
	inference_thread = threading.Thread(target=sign_inference_loop, daemon=True)
	inference_thread.start()


def stop_sign_inference():
	global inference_stop, inference_thread
	inference_stop = True
	if inference_thread:
		inference_thread.join(timeout=1.0)


def double_letter_tracking(hand_landmarks, last_x, total_x):
	# this will track double letters
	# anywhere over 0.15 of screen = double?
	curr_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
	total_x = total_x + (curr_x - last_x)

def transcribe():
	# this will transcribe letters
	dummy = "dingus"

def capture_hands():
	# this will be thing that stores all images, placeholder for now
	last_x = 0
	total_x = 0
	curr_x = 0
	double_letter = False
	IMAGES = []
	with mp_hands.Hands(
		static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
	) as hands:
		for idx, file in enumerate(IMAGES):
			# Read an image, flip it around y-axis for correct handedness output (see
			# above).
			image = cv2.flip(cv2.imread(file), 1)
			# Convert the BGR image to RGB before processing.
			results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

			# Print handedness and draw hand landmarks on the image.
			print("Handedness:", results.multi_handedness)
			if not results.multi_hand_landmarks:
				return
			image_height, image_width, _ = image.shape
			annotated_image = image.copy()
			for hand_landmarks in results.multi_hand_landmarks:
				print("hand_landmarks:", hand_landmarks)
				print(
					f"Index finger tip coordinates: (",
					f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, "
					f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})",
				)
				mp_drawing.draw_landmarks(
					annotated_image,
					hand_landmarks,
					mp_hands.HAND_CONNECTIONS,
					mp_drawing_styles.get_default_hand_landmarks_style(),
					mp_drawing_styles.get_default_hand_connections_style(),
				)
			cv2.imwrite(
				"/tmp/annotated_image" + str(idx) + ".png", cv2.flip(annotated_image, 1)
			)
			# Draw hand world landmarks.
			if not results.multi_hand_world_landmarks:
				return
			for hand_world_landmarks in results.multi_hand_world_landmarks:
				mp_drawing.plot_landmarks(
					hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5
				)
			curr_letter = "a" # placeholder value --> will store actual current letter
			last_letter = "b" # ditto
			if curr_letter != last_letter:
				if total_x >= 0.15:
					double_letter = True
				last_x = 0
				total_x = 0
				transcribe(last_letter, double_letter)
				double_letter = False
			# checks for double letters
			curr_x = results.multi_hand_landmarks[mp_hands.HandLandmark.WRIST].x
			if curr_letter == last_letter:
				double_letter_tracking(hand_landmarks, last_x, curr_x)
			curr_x = 0
