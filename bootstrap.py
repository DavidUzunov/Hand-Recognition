#!/usr/bin/env python3
"""
Bootstrap script to initialize and start the Hand Recognition application.
This is the main entry point for starting the web server and camera capture.
"""

from app import start_camera_capture
from web import app


def main():
    """Initialize camera and start the web server"""
    print("=" * 60)
    print("Starting Hand Recognition Application (Signly)")
    print("=" * 60)

    # Initialize camera capture
    print("Initializing camera capture...")
    start_camera_capture()

    # Start Flask web server
    print("Starting web server on http://0.0.0.0:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
