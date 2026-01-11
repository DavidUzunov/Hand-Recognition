#!/usr/bin/env python3
"""
Bootstrap script to initialize and start the Hand Recognition application.
This is the main entry point for starting the web server and camera capture.
"""

import signal
import sys
import os

# Disable TensorFlow's oneDNN optimizations which can cause shutdown issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# If a `.venv` exists in the workspace, automatically re-exec this script with
# that interpreter so users who run `python bootstrap.py` from system Python
# get the correct environment and avoid "module not found" errors.
VENV_PY = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
if os.path.exists(VENV_PY):
	try:
		# Compare resolved paths to avoid unnecessary re-exec
		if os.path.realpath(sys.executable) != os.path.realpath(VENV_PY):
			os.execv(VENV_PY, [VENV_PY] + sys.argv)
	except Exception:
		# Fall back to continuing with the current interpreter
		pass

import app as app_module
from web import app, socketio
import web as web_module
import os
import threading
import time
import atexit


# Exit handler to prevent TensorFlow cleanup crashes
def exit_handler():
	"""Clean exit handler"""
	try:
		import os
		os._exit(0)  # Force exit without cleanup
	except:
		pass


atexit.register(exit_handler)


def shutdown_handler(signum, frame):
	"""Handle graceful shutdown on SIGINT or SIGTERM"""
	print("\n" + "=" * 60)
	print("Shutting down Hand Recognition Application...")
	print("=" * 60)

	try:
		# Stop camera capture
		print("Stopping camera capture...")
		app_module.stop_capture = True
		if app_module.capture_thread and app_module.capture_thread.is_alive():
			app_module.capture_thread.join(timeout=2.0)

		# Stop gesture processor (daemon thread will stop automatically)
		print("Stopping gesture processor...")
		# gesture_processor thread is daemon, will stop automatically

		# Stop ping thread
		print("Stopping WebSocket ping thread...")
		web_module.stop_ping_thread = True

		# Disconnect all WebSocket clients
		print("Disconnecting WebSocket clients...")
		try:
			with app.app_context():
				socketio.emit("disconnecting", {"message": "Server shutting down"}, to=None)
		except Exception as e:
			print(f"Error emitting disconnect message: {e}")

		print("Shutdown complete. Goodbye!")
		print("=" * 60)
		time.sleep(0.5)  # Brief delay to allow cleanup
	except Exception as e:
		print(f"Error during shutdown: {e}")
		import traceback
		traceback.print_exc()
	finally:
		sys.exit(0)


def run_http():
	"""Run HTTP server"""
	socketio.run(app, host="0.0.0.0", port=5000)


def run_https():
	"""Run HTTPS server"""
	cert_file = os.environ.get("SSL_CERT", "cert.pem")
	key_file = os.environ.get("SSL_KEY", "key.pem")
	if os.path.exists(cert_file) and os.path.exists(key_file):
		ssl_context = (cert_file, key_file)
		socketio.run(
			app,
			host="0.0.0.0",
			port=5000,
			ssl_context=ssl_context,
			debug=False,
			use_reloader=False,
		)


def main():
	"""Initialize camera and start the web server"""
	print("=" * 60)
	print("Starting Hand Recognition Application (Signly)")
	print("=" * 60)

	# Register signal handlers for graceful shutdown
	signal.signal(signal.SIGINT, shutdown_handler)
	signal.signal(signal.SIGTERM, shutdown_handler)

	# Start camera capture thread (if local camera is available)
	app_module.start_camera_capture()

	# Start Flask web server with WebSocket support
	print("Starting web server on https://0.0.0.0:5000")
	print("WebSocket server ready for client connections")
	print("Press Ctrl+C to shutdown gracefully")
	print("=" * 60)

	try:
		run_https()
	except KeyboardInterrupt:
		shutdown_handler(None, None)
	except Exception as e:
		print(f"Server error: {e}")
		import traceback
		traceback.print_exc()
		shutdown_handler(None, None)


if __name__ == "__main__":
	main()
