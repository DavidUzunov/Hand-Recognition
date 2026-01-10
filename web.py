from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    Response,
    send_from_directory,
)
import os
import glob
import cv2
from app import (
    generate_frames,
    frame_buffer,
    camera_id,
    camera_available,
    start_camera_capture,
    set_camera_id,
    sign_active,
    set_sign_active,
)

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/ping")
def ping():
    return jsonify({"message": "Pong!", "status": "success"})


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


@app.route("/sign_status")
def sign_status():
    """Get current signing status"""
    return jsonify({"signing_active": sign_active})


@app.route("/toggle_sign")
def toggle_sign():
    """Toggle hand sign detection on or off"""
    new_state = request.args.get("active", type=lambda x: x.lower() == "true")
    set_sign_active(new_state)
    return jsonify(
        {
            "status": "success",
            "signing_active": sign_active,
            "message": f"Hand sign detection {'started' if sign_active else 'stopped'}",
        }
    )


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


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with custom page"""
    return render_template("404.html"), 404


# Flask app will be started by bootstrap.py
