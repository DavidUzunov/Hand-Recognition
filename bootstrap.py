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
		socketio.run(app, debug=True, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
	except KeyboardInterrupt:
		shutdown_handler(None, None)

if __name__ == "__main__":
	main()
