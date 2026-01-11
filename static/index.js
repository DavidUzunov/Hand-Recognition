// Import common logic
import './common.js';

function saveTranscript() {
	const messageBox = document.getElementById('message-box');
	if (!messageBox || !messageBox.value) {
		console.log('No transcript to save');
		return;
	}
	const blob = new Blob([messageBox.value], { type: 'text/plain' });
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = 'translation.txt';
	document.body.appendChild(a);
	a.click();
	setTimeout(() => {
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	}, 100);
}
const socket = io();

// Metrics tracking
let frameCount = 0;
let lastFrameTime = Date.now();
let lastPingTime = null;
let fpsUpdateInterval = null;

socket.on('connect', () => {
	console.log('Connected to server');
	document.getElementById('socket-status').textContent = 'Connected';
	document.getElementById('socket-status').style.color = '#4ade80';
});

// New ping logic: client sends timestamp, server echoes, client calculates RTT
let lastPingSent = null;
function sendClientPing() {
	lastPingSent = performance.now();
	socket.emit('client_ping', { t0: lastPingSent });
}

socket.on('client_pong', (data) => {
	const t0 = data.t0;
	const now = performance.now();
	if (typeof t0 === 'number') {
		const rtt = now - t0;
		document.getElementById('ping-time').textContent = `${rtt.toFixed(1)}ms`;
	} else {
		document.getElementById('ping-time').textContent = 'N/A';
	}
});

// Optionally, ping every 2s for live display
setInterval(sendClientPing, 2000);

socket.on('camera_status', (data) => {
	console.log('Received camera status:', data);
	updateCameraStatus(data);
	const cameraHeader = document.getElementById('camera-status-header');
	const selectedCamera = document.getElementById('selected-camera');
	if (cameraHeader) {
		cameraHeader.textContent = data.camera_available ? 'Available' : 'Unavailable';
	}
	if (selectedCamera) {
		selectedCamera.textContent = data.device || '-';
	}
});

// Update transcript when receiving ASL transcript messages
socket.on('asl_transcript', (data) => {
	if (data && data.message) {
		console.log('[asl_transcript] Received:', data.message);
		const messageBox = document.getElementById('message-box');
		if (messageBox) {
			messageBox.value += data.message;
		}
	} else {
		console.log('[asl_transcript] Received empty or malformed data:', data);
	}
});
