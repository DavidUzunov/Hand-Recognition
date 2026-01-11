from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import glob

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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

# Global variables for video capture
cap = None
current_frame = None  # JPG
frame_lock = threading.Lock()
camera_available = False
camera_id = 0  # Default camera ID
capture_thread = None
stop_capture = False
sign_active = False  # Track if hand signing capture is active

# Global variables for hand-tracking/transcribing
curr_word = ""
last_x = 0
total_x = 0
curr_x = 0
double_letter = False
curr_letter = ""
last_letter = ""


def detect_available_cameras():
    """Detect available camera devices and return list of camera IDs"""
    available_cameras = []
    # Check /dev/video* devices on Linux
    for video_device in sorted(glob.glob("/dev/video*")):
        try:
            device_num = int(video_device.split("video")[1])
            cap = cv2.VideoCapture(device_num, cv2.CAP_V4L2)
            if cap.isOpened():
                available_cameras.append(device_num)
                cap.release()
        except (ValueError, IndexError):
            pass
    return available_cameras


def set_default_camera():
    """Set camera_id to first available camera, or 0 if none available"""
    global camera_id
    available = detect_available_cameras()
    if available:
        camera_id = available[0]
        print(
            f"Found {len(available)} camera(s). Using camera {camera_id} (/dev/video{camera_id})"
        )
        if len(available) > 1:
            print(
                f"Other cameras available: {[f'/dev/video{c}' for c in available[1:]]}"
            )
    else:
        camera_id = 0
        print("No cameras detected. Using default camera_id=0")


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


def capture_frames():
    """Continuously capture frames from USB webcam (Raspberry Pi compatible)"""
    global cap, current_frame, camera_available, camera_id, stop_capture

    # Try to open USB camera with selected ID
    cap = cv2.VideoCapture(
        camera_id, cv2.CAP_V4L2
    )  # Use V4L2 backend for better Raspberry Pi USB camera support

    # Give the camera time to initialize
    time.sleep(0.5)

    # Check if camera is available
    if not cap.isOpened():
        print(
            f"WARNING: No USB camera detected on /dev/video{camera_id}. Using default image."
        )
        camera_available = False
        with frame_lock:
            current_frame = default_image.copy()
        cap.release()
    else:
        camera_available = True
        # Set camera properties optimized for Raspberry Pi USB cameras
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        print("USB camera detected and initialized successfully on Raspberry Pi.")

    while True:
        if camera_available:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera. Switching to default image.")
                camera_available = False
                with frame_lock:
                    current_frame = default_image.copy()
                break

            # Check if we should stop capturing
            if stop_capture:
                print("Stopping camera capture...")
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
            time.sleep(0.1)

    if cap is not None:
        cap.release()


def generate_frames():
    """Generate frames for streaming and optionally process hands"""
    hands = get_hands_detector()

    while True:
        with frame_lock:
            # Use default image if current_frame is None or camera is not available
            frame_to_send = (
                current_frame if current_frame is not None else default_image
            )
            frame_to_process = frame_to_send.copy()
            is_default_image = not camera_available

        # Process hand detection if signing is active
        if sign_active and frame_to_process is not None and not is_default_image:
            try:
                # Convert BGR to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # Draw hand landmarks if detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_to_process,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )
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


def transcribe(letter, double_letter, send):
    # this will transcribe letters
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
    # normalizes the data via the wrist for better model processing
    for measurement in hand.landmark:
        data_list.append(measurement.x - hand.landmark[0].x)
        data_list.append(measurement.y - hand.landmark[0].y)
        data_list.append(measurement.z - hand.landmark[0].z)
    return np.array(data_list).reshape(1, -1)


def capture_hands(curr_image):
    # this will be thing that stores all images, placeholder for now
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(curr_image), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return
        primary_hand = results.multi_hand_landmarks[0]
        data = process_hand_data(primary_hand)
        send = False
        if curr_letter != last_letter:
            if total_x >= 0.15:
                double_letter = True
            last_x = 0
            total_x = 0
            if curr_letter == " ":
                send = True
            transcribe(last_letter, double_letter, send)
            double_letter = False
            last_letter = curr_letter
        # checks for double letters via wrist data
        if curr_letter == last_letter:
            curr_x = primary_hand.landmark[0].x
            total_x = total_x + (curr_x - last_x)
            last_x = curr_x
        curr_x = 0
