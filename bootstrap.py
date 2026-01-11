#!/usr/bin/env python3
"""
Bootstrap script to initialize and start the Hand Recognition application.
This is the main entry point for starting the web server and camera capture.
"""

import signal
import sys
import app as app_module
from app import start_camera_capture
from web import app, socketio
import web as web_module
import os
from flask import request, redirect
import ssl


def shutdown_handler(signum, frame):
    """Handle graceful shutdown on SIGINT or SIGTERM"""
    print("\n" + "=" * 60)
    print("Shutting down Hand Recognition Application...")
    print("=" * 60)

    # Stop camera capture
    print("Stopping camera capture...")
    app_module.stop_capture = True

    # Stop ping thread
    print("Stopping WebSocket ping thread...")
    web_module.stop_ping_thread = True

    # Disconnect all WebSocket clients
    print("Disconnecting WebSocket clients...")
    socketio.emit("disconnecting", {"message": "Server shutting down"}, to=None)

    print("Shutdown complete. Goodbye!")
    print("=" * 60)
    sys.exit(0)


# --- SSL/Redirect logic ---
@app.before_request
def force_ssl_or_http():
    # Only allow HTTPS for /host, force HTTP for all others
    if request.endpoint == "host":
        if request.scheme != "https":
            # Redirect to HTTPS
            url = request.url.replace("http://", "https://", 1)
            return redirect(url, code=301)
    else:
        if request.scheme == "https":
            # Redirect to HTTP
            url = request.url.replace("https://", "http://", 1)
            return redirect(url, code=301)


def main():
    """Initialize camera and start the web server"""
    print("=" * 60)
    print("Starting Hand Recognition Application (Signly)")
    print("=" * 60)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Auto-detect available cameras and set default
    print("Detecting available cameras...")
    app_module.set_default_camera()

    # Initialize camera capture
    print("Initializing camera capture...")
    start_camera_capture()

    # Start Flask web server with WebSocket support
    print("Starting web server on http://0.0.0.0:5000")
    print("WebSocket server ready for client connections")
    print("Press Ctrl+C to shutdown gracefully")
    print("=" * 60)

    try:
        ssl_context = None
        # Only enable SSL if certs are present
        cert_file = os.environ.get("SSL_CERT", "cert.pem")
        key_file = os.environ.get("SSL_KEY", "key.pem")
        if os.path.exists(cert_file) and os.path.exists(key_file):
            ssl_context = (cert_file, key_file)
        socketio.run(
            app,
            debug=True,
            host="0.0.0.0",
            port=5000,
            allow_unsafe_werkzeug=True,
            use_reloader=True,
            ssl_context=ssl_context,
        )
    except KeyboardInterrupt:
        shutdown_handler(None, None)


if __name__ == "__main__":
    main()
