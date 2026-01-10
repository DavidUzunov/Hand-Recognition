from flask import Flask, render_template, jsonify, request
from collections import deque
import ssl

app = Flask(__name__)

# Memory buffer to store frames (max 30 frames)
frame_buffer = deque(maxlen=30)


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/ping")
def ping():
    return jsonify({"message": "Pong!", "status": "success"})


@app.route("/camera")
def camera():
    return render_template("camera.html")


# test for webcam input
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


if __name__ == "__main__":
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain("cert.pem", "key.pem")
    app.run(debug=True, ssl_context=ssl_context)
