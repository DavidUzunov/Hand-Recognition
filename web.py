# --- Imports ---
import os
import glob
import cv2
import threading
import time
from functools import wraps
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    Response,
    send_from_directory,
    redirect,
    url_for,
)
from flask_socketio import SocketIO, emit
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

# --- App Setup ---
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
connected_clients = 0
ping_thread = None
stop_ping_thread = False


# --- Utility & Auth Helpers ---
def check_auth(username, password):
    return username == "admin" and password == "admin"


def authenticate():
    return Response(
        "Could not verify your access level for that URL.\n"
        "You have to login with proper credentials",
        401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'},
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


def send_asl_transcript(message):
    if connected_clients > 0:
        with app.app_context():
            socketio.emit("asl_transcript", {"message": message}, to=None)


def send_ping_messages():
    global stop_ping_thread
    while not stop_ping_thread:
        if connected_clients > 0:
            with app.app_context():
                socketio.emit(
                    "ping", {"message": "ping", "timestamp": time.time()}, to=None
                )
        time.sleep(2)


def get_camera_status():
    available_cameras = []
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


set_asl_transcript_callback(send_asl_transcript)


# --- Socket Handlers ---
@socketio.on("host_frame")
def handle_host_frame(frame_bytes):
    print(f"Received host frame: {len(frame_bytes)} bytes")
    pass


@socketio.on("get_sign_status")
def handle_get_sign_status():
    with app.app_context():
        emit(
            "sign_status",
            {"signing_active": sign_active},
        )


@socketio.on("toggle_sign")
def handle_toggle_sign(data=None):
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


@socketio.on("client_ping")
def handle_client_ping(data):
    t0 = data.get("t0") if data else None
    emit("client_pong", {"t0": t0, "server_ts": time.time()})


@socketio.on("connect")
def handle_connect():
    global connected_clients, ping_thread, stop_ping_thread
    connected_clients += 1
    print(f"Client connected. Total clients: {connected_clients}")
    if ping_thread is None or not ping_thread.is_alive():
        stop_ping_thread = False
        ping_thread = threading.Thread(target=send_ping_messages, daemon=True)
        ping_thread.start()
    emit("connected", {"message": "Connected to server"})
    camera_status = get_camera_status()
    emit("camera_status", camera_status)


@socketio.on("disconnect")
def handle_disconnect():
    global connected_clients
    connected_clients -= 1
    print(f"Client disconnected. Total clients: {connected_clients}")


# --- Host Endpoint ---
@app.route("/host")
def host():
    return render_template("host.html")


# --- Admin Endpoint ---
@app.route("/admin", methods=["GET", "POST"])
@requires_auth
def admin():
    available_cameras = []
    for video_device in sorted(glob.glob("/dev/video*")):
        try:
            device_num = int(video_device.split("video")[1])
            cap = cv2.VideoCapture(device_num, cv2.CAP_V4L2)
            if cap.isOpened():
                available_cameras.append(device_num)
                cap.release()
        except (ValueError, IndexError):
            pass
    status = None
    debug_data = None
    if request.method == "POST":
        try:
            new_id = int(request.form.get("camera_id"))
            set_camera_id(new_id)
            start_camera_capture()
            status = f"Camera switched to /dev/video{new_id}"
        except Exception as e:
            status = f"Error: {e}"
    # Get debug info for admin page
    debug_data = None
    try:
        import threading as thread_module

        camera_info = get_camera_status()
        frame_buffer_info = {
            "size": len(frame_buffer),
            "max_size": frame_buffer.maxlen,
        }
        threads_info = {
            "total_threads": thread_module.active_count(),
            "current_thread": thread_module.current_thread().name,
            "all_threads": [t.name for t in thread_module.enumerate()],
        }
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
        websocket_info = {
            "connected_clients": connected_clients,
            "ping_thread_running": (
                ping_thread is not None and ping_thread.is_alive()
                if ping_thread
                else False
            ),
            "stop_ping_thread_flag": stop_ping_thread,
        }
        signing_info = {
            "signing_active": sign_active,
        }
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
    except Exception as e:
        debug_data = {"error": str(e)}
    return render_template(
        "admin.html",
        available_cameras=available_cameras,
        current_camera=camera_id,
        status=status,
        debug_data=debug_data,
    )


# --- Main Routes ---
@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/ping")
def ping():
    return jsonify({"message": "Pong!", "status": "success"})


@app.route("/set_camera")
def set_camera():
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
    return jsonify(get_camera_status())


@app.route("/no_camera")
def no_camera():
    try:
        import io

        img = create_default_image()
        _, png = cv2.imencode(".png", img)
        return Response(png.tobytes(), mimetype="image/png")
    except Exception as e:
        print(f"Error generating no_camera image: {e}")
        return "Camera not available", 503


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/test")
def test():
    pass


@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    if "frame" not in request.files:
        return jsonify({"error": "No frame provided"}), 400
    frame_file = request.files["frame"]
    frame_data = frame_file.read()
    frame_buffer.append(frame_data)
    print(f"Got image - Buffer size: {len(frame_buffer)}")
    return jsonify({"status": "success", "buffer_size": len(frame_buffer)})


# --- Error Handlers ---
@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


# Flask app will be started by bootstrap.py
