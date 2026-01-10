from flask import Flask, render_template, jsonify
import ssl

app = Flask(__name__)


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


if __name__ == "__main__":
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain("cert.pem", "key.pem")
    app.run(debug=True, ssl_context=ssl_context)
