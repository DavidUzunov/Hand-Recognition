from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    Response,
    send_from_directory,
)
from flask_socketio import SocketIO, emit
import os
import glob
import cv2
import threading
import time
from app import (
    generate_frames,
    frame_buffer,
    camera_id,
    camera_available,
    start_camera_capture,
    set_camera_id,
    sign_active,
    set_sign_active,
    create_default_image,
    set_asl_transcript_callback,
)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Track connected clients and ping thread
connected_clients = 0
ping_thread = None
stop_ping_thread = False


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/ping")
def ping():
    return jsonify({"message": "Pong!", "status": "success"})


def send_asl_transcript(message):
    """Send an ASL transcript message to all connected clients via WebSocket"""
    if connected_clients > 0:
        with app.app_context():
            socketio.emit("asl_transcript", {"message": message}, to=None)


# Set the callback to avoid circular import
set_asl_transcript_callback(send_asl_transcript)


@app.route("/set_camera")
def set_camera():
    """Set camera ID via query parameter (?id=0, ?id=1, etc.)"""
    new_id = request.args.get("id", type=int)
    if new_id is None:
        return (
            jsonify(
                {
                    "error": "Camera ID not provided. Use ?id=0, ?id=1, etc.",
                    "current_camera": camera_id,
                }
            ),
            400,
        )

    if new_id < 0:
        return jsonify({"error": "Camera ID must be >= 0"}), 400

    old_id = camera_id
    set_camera_id(new_id)
    print(f"Switching camera from {old_id} to {camera_id}")

    # Restart camera capture with new ID
    start_camera_capture()

    return jsonify(
        {
            "status": "success",
            "message": f"Camera switched to /dev/video{camera_id}",
            "previous_camera": old_id,
            "current_camera": camera_id,
        }
    )


@app.route("/camera_status")
def camera_status():
    """Get current camera status and list available camera IDs"""
    # Detect available camera devices
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

    return jsonify(
        {
            "camera_id": camera_id,
            "camera_available": camera_available,
            "device": f"/dev/video{camera_id}",
            "available_cameras": available_cameras,
        }
    )


@app.route("/no_camera")
def no_camera():
    """Serve the default no-camera image"""
    try:
        import io

        # Use the default image from app.py
        img = create_default_image()

        # Convert image to PNG bytes
        _, png = cv2.imencode(".png", img)

        return Response(png.tobytes(), mimetype="image/png")
    except Exception as e:
        print(f"Error generating no_camera image: {e}")
        return "Camera not available", 503


@app.route("/favicon.ico")
def favicon():
    """Serve the favicon"""
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


@app.route("/video_feed")
def video_feed():
    """Stream video frames"""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/test")
def test():
    pass


@socketio.on("get_sign_status")
def handle_get_sign_status():
    """Handle sign status request via WebSocket"""
    with app.app_context():
        emit(
            "sign_status",
            {"signing_active": sign_active},
        )


@socketio.on("toggle_sign")
def handle_toggle_sign(data):
    """Handle sign toggle request via WebSocket"""
    if data is None:
        data = {}
    new_state = data.get("active", not sign_active)
    set_sign_active(new_state)
    with app.app_context():
        emit(
            "sign_status",
            {"signing_active": sign_active},
            to=None,
        )


@app.route("/debug")
def debug():
    """Comprehensive debug endpoint that returns info about everything"""
    import threading as thread_module

    # Get camera info
    camera_info = get_camera_status()

    # Get frame buffer info
    frame_buffer_info = {
        "size": len(frame_buffer),
        "max_size": frame_buffer.maxlen,
    }

    # Get thread info
    threads_info = {
        "total_threads": thread_module.active_count(),
        "current_thread": thread_module.current_thread().name,
        "all_threads": [t.name for t in thread_module.enumerate()],
    }

    # Get capture thread info
    from app import (
        capture_thread as app_capture_thread,
        stop_capture as app_stop_capture,
    )

    capture_thread_info = {
        "thread_running": (
            app_capture_thread is not None and app_capture_thread.is_alive()
            if app_capture_thread
            else False
        ),
        "stop_capture_flag": app_stop_capture,
    }

    # Get WebSocket info
    websocket_info = {
        "connected_clients": connected_clients,
        "ping_thread_running": (
            ping_thread is not None and ping_thread.is_alive() if ping_thread else False
        ),
        "stop_ping_thread_flag": stop_ping_thread,
    }

    # Get signing info
    signing_info = {
        "signing_active": sign_active,
    }

    # Compile all debug info
    debug_data = {
        "status": "ok",
        "timestamp": time.time(),
        "camera": camera_info,
        "frame_buffer": frame_buffer_info,
        "threads": threads_info,
        "capture_thread": capture_thread_info,
        "websocket": websocket_info,
        "signing": signing_info,
    }

    return jsonify(debug_data)


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


def send_ping_messages():
    """Send ping messages every 2 seconds to all connected clients"""
    global stop_ping_thread
    while not stop_ping_thread:
        if connected_clients > 0:
            with app.app_context():
                socketio.emit(
                    "ping", {"message": "ping", "timestamp": time.time()}, to=None
                )
        time.sleep(2)


def send_asl_transcript(message):
    """Send an ASL transcript message to all connected clients via WebSocket"""
    if connected_clients > 0:
        with app.app_context():
            socketio.emit("asl_transcript", {"message": message}, to=None)


def get_camera_status():
    """Get current camera status to send via WebSocket"""
    # Detect available camera devices
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

    return {
        "camera_id": camera_id,
        "camera_available": camera_available,
        "device": f"/dev/video{camera_id}" if camera_available else None,
        "available_cameras": available_cameras,
    }


@socketio.on("connect")
def handle_connect():
    """Handle client connection"""
    global connected_clients, ping_thread, stop_ping_thread
    connected_clients += 1
    print(f"Client connected. Total clients: {connected_clients}")

    # Start ping thread if not already running
    if ping_thread is None or not ping_thread.is_alive():
        stop_ping_thread = False
        ping_thread = threading.Thread(target=send_ping_messages, daemon=True)
        ping_thread.start()

    emit("connected", {"message": "Connected to server"})

    # Send camera status on connect
    camera_status = get_camera_status()
    emit("camera_status", camera_status)


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""
    global connected_clients
    connected_clients -= 1
    print(f"Client disconnected. Total clients: {connected_clients}")


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with custom page"""
    return render_template("404.html"), 404


# Flask app will be started by bootstrap.py
