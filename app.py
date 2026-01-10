from flask import Flask, render_template, jsonify, request, Response
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import threading
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


app = Flask(__name__)

# Memory buffer to store frames (max 30 frames)
frame_buffer = deque(maxlen=30)

# Global variables for video capture
cap = None
current_frame = None
frame_lock = threading.Lock()
camera_available = False


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

	# Add a camera icon using shapes
	icon_center_x, icon_center_y = 640, 240
	cv2.rectangle(
		img,
		(icon_center_x - 60, icon_center_y - 40),
		(icon_center_x + 60, icon_center_y + 40),
		(150, 150, 150),
		3,
	)
	cv2.circle(img, (icon_center_x, icon_center_y), 25, (150, 150, 150), 3)
	cv2.line(
		img,
		(icon_center_x + 60, icon_center_y - 40),
		(icon_center_x + 80, icon_center_y - 60),
		(150, 150, 150),
		3,
	)
	cv2.line(
		img,
		(icon_center_x + 80, icon_center_y - 60),
		(icon_center_x + 80, icon_center_y - 30),
		(150, 150, 150),
		3,
	)
	cv2.line(
		img,
		(icon_center_x + 80, icon_center_y - 30),
		(icon_center_x + 60, icon_center_y - 10),
		(150, 150, 150),
		3,
	)

	return img


# Generate default image once at startup
default_image = create_default_image()


def capture_frames():
	"""Continuously capture frames from webcam"""
	global cap, current_frame, camera_available
	cap = cv2.VideoCapture(0)

	# Check if camera is available
	if not cap.isOpened():
		print("WARNING: No camera detected. Using default image.")
		camera_available = False
		with frame_lock:
			current_frame = default_image.copy()
		cap.release()
	else:
		camera_available = True
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
		print("Camera detected and initialized successfully.")

	while True:
		if camera_available:
			ret, frame = cap.read()
			if not ret:
				print("Failed to read frame from camera. Switching to default image.")
				camera_available = False
				with frame_lock:
					current_frame = default_image.copy()
				break
			with frame_lock:
				current_frame = frame.copy()
				# Store frame in buffer
				_, jpeg = cv2.imencode(".jpg", frame)
				frame_buffer.append(jpeg.tobytes())
				print(f"Got image - Buffer size: {len(frame_buffer)}")
		else:
			# If no camera, just keep the default image
			with frame_lock:
				current_frame = default_image.copy()

	if cap is not None:
		cap.release()


def generate_frames():
	"""Generate frames for streaming"""
	while True:
		with frame_lock:
			# Use default image if current_frame is None or camera is not available
			frame_to_send = (
				current_frame if current_frame is not None else default_image
			)

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

		# Add a small delay to prevent excessive CPU usage
		time.sleep(0.033)  # ~30 fps


# Start frame capture in background thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

def capture_hands():
    dummy = "doofus"
    with mp_hands.Hands(
        static_image_mode=True,
    )


@app.route("/")
def hello_world():
	return render_template("index.html")


@app.route("/ping")
def ping():
	return jsonify({"message": "Pong!", "status": "success"})


@app.route("/favicon.ico")
def favicon():
	"""Serve the favicon"""
	from flask import send_from_directory
	import os

	return send_from_directory(
		os.path.join(app.root_path, "static"),
		"favicon.ico",
		mimetype="image/vnd.microsoft.icon",
	)


@app.route("/camera")
def camera():
	return render_template("camera.html")


@app.route("/video_feed")
def video_feed():
	"""Stream video frames"""
	return Response(
		generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
	)


@app.route("/test")
def test():

	pass


@app.route("/upload_frame", methods=["POST"])
def upload_frame():
	"""Receive camera frame and store in memory buffer"""
	if "frame" not in request.files:
		return jsonify({"error": "No frame provided"}), 400

	frame_file = request.files["frame"]
	frame_data = frame_file.read()

	# Store frame in buffer
	frame_buffer.append(frame_data)

	# Print message indicating frame received
	print(f"Got image - Buffer size: {len(frame_buffer)}")

	return jsonify({"status": "success", "buffer_size": len(frame_buffer)})


if app.__name__ == "__main__":
	app.run(debug=True, host="0.0.0.0", port=5000)
