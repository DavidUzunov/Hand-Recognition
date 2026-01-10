from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import threading
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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


def double_letter_tracking(hand_landmarks, last_x, total_x):
    # this will track double letters
	# anywhere over 0.15 of screen = double?
    dummy = "doofus"
    curr_x= hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x



def capture_hands():
    # this will be thing that stores all images, placeholder for now
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


# Camera capture will be started by bootstrap.py
