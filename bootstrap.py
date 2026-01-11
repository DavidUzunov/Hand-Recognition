#!/usr/bin/env python3
"""
Bootstrap script to initialize and start the Hand Recognition application.
This is the main entry point for starting the web server and camera capture.
"""

import signal
import sys
import app as app_module
from web import app, socketio
import web as web_module
import os
import threading
from multiprocessing import Process


def shutdown_handler(signum, frame):
	global p1
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

	# Terminate camera process
	print("Terminating camera process...")
	p1.terminate()

	# Disconnect all WebSocket clients
	print("Disconnecting WebSocket clients...")
	socketio.emit("disconnecting", {"message": "Server shutting down"}, to=None)

	print("Shutdown complete. Goodbye!")
	print("=" * 60)
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
	global p1
	"""Initialize camera and start the web server"""
	print("=" * 60)
	print("Starting Hand Recognition Application (Signly)")
	print("=" * 60)

	# Register signal handlers for graceful shutdown
	signal.signal(signal.SIGINT, shutdown_handler)
	signal.signal(signal.SIGTERM, shutdown_handler)

	# Camera logic removed: only host stream is used

	# Start Flask web server with WebSocket support
	print("Starting web server on https://0.0.0.0:5000")
	print("WebSocket server ready for client connections")
	print("Press Ctrl+C to shutdown gracefully")
	print("=" * 60)

	run_https()
	p1 = Process(target=app_module.capture_hands, args=(app_module.frame_byte_q,))
	p1.start()


if __name__ == "__main__":
	main()
