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
    frame_buffer,
    sign_active,
    set_sign_active,
    create_default_image,
    set_asl_transcript_callback,
    capture_hands,
)

# --- App Setup ---
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
connected_clients = 0
ping_thread = None
stop_ping_thread = False

# Track if a host is logged in
host_logged_in = False


# --- Utility & Auth Helpers ---
def check_auth(username, password, role="admin"):
    if role == "admin":
        return username == "admin" and password == "admin"
    elif role == "host":
        return username == "host" and password == "host"
    return False


def authenticate(role="admin"):
    realm = "Host Login" if role == "host" else "Login Required"
    resp = Response(render_template("401.html"), 401)
    resp.headers["WWW-Authenticate"] = f'Basic realm="{realm}"'
    return resp


def requires_auth(role="admin"):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            auth = request.authorization
            if not auth or not check_auth(auth.username, auth.password, role=role):
                return authenticate(role=role)
            return f(*args, **kwargs)

        return decorated

    return decorator


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
    # Only host stream is available
    return {
        "camera_id": None,
        "camera_available": len(frame_buffer) > 0,
        "device": "host_stream" if len(frame_buffer) > 0 else None,
        "available_cameras": [],
    }


set_asl_transcript_callback(send_asl_transcript)


# --- Socket Handlers ---
@socketio.on("host_frame")
def handle_host_frame(frame_bytes):
    # Store host stream frames in the same buffer as before
    frame_buffer.append(frame_bytes)
    if sign_active == True:
        capture_hands(frame_bytes)
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
    global connected_clients, host_logged_in
    connected_clients -= 1
    print(f"Client disconnected. Total clients: {connected_clients}")
    # If no clients are connected, clear host_logged_in
    if connected_clients <= 0:
        host_logged_in = False


# --- Host Endpoint ---
@app.route("/host")
def host():
    key = request.args.get("key")
    if key != "123":
        return render_template("401.html"), 401
    # Only redirect if camera is actively streaming (frame_buffer has frames)
    if len(frame_buffer) > 0:
        return redirect(url_for("admin", key=key))
    return render_template("host.html")


# --- Admin Endpoint ---
@app.route("/admin")
def admin():
    key = request.args.get("key")
    if key != "123":
        return render_template("401.html"), 401
    status = None
    debug_data = None
    # Get debug info for admin page
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
            "websocket": websocket_info,
            "signing": signing_info,
        }
    except Exception as e:
        debug_data = {"error": str(e)}
    return render_template(
        "admin.html",
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


# Serve the latest host frame as a single JPEG, fallback to no_camera
@app.route("/video_feed")
def video_feed():
    def gen():
        import time

        boundary = b"--frame\r\n"
        while True:
            if len(frame_buffer) > 0:
                frame = frame_buffer[-1]
                yield boundary
                yield b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            else:
                # Wait for a frame
                time.sleep(0.1)

    if len(frame_buffer) == 0:
        # No host stream, fallback to no_camera
        return no_camera()
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/test")
def test():
    pass


# --- Error Handlers ---
@app.errorhandler(401)
def unauthorized(error):
    return render_template("401.html"), 401


@app.errorhandler(403)
def forbidden(error):
    return render_template("403.html"), 403


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404
