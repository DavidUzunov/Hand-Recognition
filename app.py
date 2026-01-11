"""
Hand Recognition Application - Frame Capture and Streaming
Main responsibility: Capture frames from camera/host and stream them via HTTP/WebSocket
"""

from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import glob

# Try to load mediapipe solutions (may not be available in some versions)
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


# ============================================================================
# GLOBAL STATE (frame buffer for web server)
# ============================================================================

# Memory buffer to store frames (max 30 frames as JPEG bytes)
frame_buffer = deque(maxlen=30)

# Camera state variables
cap = None
current_frame = None
camera_available = False
camera_id = 0
stop_capture = False
frame_lock = threading.Lock()
capture_thread = None

# Sign active flag - whether to process hand signs
sign_active = True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


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


# ============================================================================
# FRAME CAPTURE (from local camera or WebSocket host stream)
# ============================================================================


def capture_frames():
	"""Continuously capture frames from USB webcam (Raspberry Pi compatible)"""
	global cap, current_frame, camera_available, camera_id, stop_capture

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
				print(f"Got image - Buffer size: {len(frame_buffer)}")
		else:
			with frame_lock:
				current_frame = default_image.copy()
			time.sleep(0.1)

	if cap is not None:
		try:
			cap.release()
		except Exception:
			pass


def generate_frames():
	"""Generate frames for streaming and optionally process hands for visualization"""
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
		# `mp.solutions` API is available. This draws hand landmarks on the frame.
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
	"""Set whether hand sign detection/visualization is active"""
	global sign_active
	sign_active = active
