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
		cameraHeader.style.color = data.camera_available ? '#4ade80' : '#ef4444';
	}
	if (selectedCamera && data.camera_available) {
		selectedCamera.textContent = `/dev/video${data.camera_id}`;
	}
});

socket.on('sign_status', (data) => {
	console.log('Received sign status:', data);
	updateSignButton(data.signing_active);
	const signingStatus = document.getElementById('signing-status');
	if (signingStatus) {
		signingStatus.textContent = data.signing_active ? 'Active' : 'Inactive';
		signingStatus.style.color = data.signing_active ? '#4ade80' : '#fbbf24';
	}
});

socket.on('asl_transcript', (data) => {
	console.log('Received ASL transcript:', data.message);
	const messageBox = document.getElementById('message-box');
	if (messageBox) {
		const timestamp = new Date().toLocaleTimeString();
		messageBox.value += `[${timestamp}] ${data.message}\n`;
		// Auto-scroll to bottom
		messageBox.scrollTop = messageBox.scrollHeight;
		// Update line count
		const lines = messageBox.value.split('\n').filter(line => line.trim()).length;
		document.getElementById('transcript-lines').textContent = lines;
	}
});

function updateCameraStatus(data) {
	try {
		const videoFeed = document.getElementById('video-feed');

		// Load video feed when camera is available, otherwise show placeholder
		if (data.camera_available) {
			videoFeed.src = '/video_feed';
			// Start FPS tracking when video feed loads
			startFpsTracking();
		} else {
			videoFeed.src = '/no_camera';
			stopFpsTracking();
		}

		// Populate camera selector with available cameras
		const cameraSelect = document.getElementById('camera-select');
		cameraSelect.innerHTML = '';

		if (data.available_cameras && data.available_cameras.length > 0) {
			data.available_cameras.forEach(cameraId => {
				const option = document.createElement('option');
				option.value = cameraId;
				option.textContent = `/dev/video${cameraId}`;
				if (cameraId === data.camera_id) {
					option.selected = true;
				}
				cameraSelect.appendChild(option);
			});
		} else {
			cameraSelect.innerHTML = '<option value="">No cameras available</option>';
		}
	} catch (error) {
		console.error('Error updating camera status:', error);
		document.getElementById('camera-select').innerHTML = '<option value="">Error loading cameras</option>';
	}
}

function startFpsTracking() {
	if (fpsUpdateInterval) return;
	frameCount = 0;
	lastFrameTime = Date.now();
	fpsUpdateInterval = setInterval(() => {
		const now = Date.now();
		const elapsed = (now - lastFrameTime) / 1000;
		const fps = Math.round(frameCount / elapsed);
		document.getElementById('fps-counter').textContent = fps;
		frameCount = 0;
		lastFrameTime = now;
	}, 1000);

	// Track video loads
	const videoFeed = document.getElementById('video-feed');
	if (videoFeed) {
		videoFeed.addEventListener('load', () => {
			frameCount++;
			document.getElementById('last-frame-time').textContent = new Date().toLocaleTimeString();
		});
	}
}

function stopFpsTracking() {
	if (fpsUpdateInterval) {
		clearInterval(fpsUpdateInterval);
		fpsUpdateInterval = null;
	}
	document.getElementById('fps-counter').textContent = '0';
	document.getElementById('last-frame-time').textContent = '-';
}

async function changeCamera() {
	const cameraSelect = document.getElementById('camera-select');
	const cameraId = cameraSelect.value;

	if (!cameraId) {
		return;
	}

	try {
		const response = await fetch(`/set_camera?id=${cameraId}`);
		const data = await response.json();

		if (data.status === 'success') {
			console.log(`Camera switched to ${data.current_camera}`);
			// Refresh camera status after switching
			setTimeout(getCameraStatus, 500);
		} else {
			alert(`Error: ${data.error}`);
			// Reset selector to current camera
			getCameraStatus();
		}
	} catch (error) {
		console.error('Error changing camera:', error);
		alert('Failed to change camera');
		// Reset selector to current camera
		getCameraStatus();
	}
}

async function pingServer() {
	const startTime = performance.now();

	try {
		const response = await fetch('/ping');
		const data = await response.json();
		const endTime = performance.now();
		const pingTime = (endTime - startTime).toFixed(2);

		alert(`Ping: ${pingTime}ms`);
	} catch (error) {
		alert('Error: Could not reach server');
	}
}

async function getSignStatus() {
	try {
		socket.emit('get_sign_status');
	} catch (error) {
		console.error('Error fetching sign status:', error);
	}
}

function updateSignButton(isActive) {
	const button = document.getElementById('sign-button');
	const sidebar = document.getElementById('transcript-sidebar');
	const container = document.querySelector('.container');

	if (isActive) {
		button.textContent = 'Stop Signing';
		button.classList.add('active');
		if (sidebar) sidebar.style.display = 'flex';
		if (container) container.classList.add('expanded');
	} else {
		button.textContent = 'Start Signing';
		button.classList.remove('active');
		if (sidebar) sidebar.style.display = 'none';
		if (container) container.classList.remove('expanded');
	}
}

async function toggleSign() {
	const button = document.getElementById('sign-button');
	const isCurrentlyActive = button.classList.contains('active');
	const newState = !isCurrentlyActive;

	try {
		socket.emit('toggle_sign', { active: newState });
	} catch (error) {
		console.error('Error toggling sign status:', error);
		alert('Failed to toggle signing');
	}
}

function clearTranscript() {
	const messageBox = document.getElementById('message-box');
	if (messageBox) {
		messageBox.value = '';
		document.getElementById('transcript-lines').textContent = '0';
	}
}

async function shareTranscript() {
	const messageBox = document.getElementById('message-box');
	if (!messageBox || !messageBox.value) {
		console.log('No transcript to share');
		return;
	}

	try {
		if (navigator.share) {
			// Use Web Share API if available
			await navigator.share({
				title: 'ASL Transcript',
				text: messageBox.value
			});
			console.log('Transcript shared successfully');
		} else if (navigator.clipboard) {
			// Fallback: Copy to clipboard
			await navigator.clipboard.writeText(messageBox.value);
			console.log('Transcript copied to clipboard');
		} else {
			// Fallback: Log message
			console.warn('Web Share API not supported. Please copy the text manually.');
		}
	} catch (error) {
		console.error('Error sharing transcript:', error);
	}
}

// Check for debug mode in URL
function checkDebugMode() {
	const params = new URLSearchParams(window.location.search);
	if (params.has('debug')) {
		const pingButton = document.getElementById('ping-button');
		if (pingButton) {
			pingButton.style.display = 'block';
		}
	}
}

// Load sign status when page loads
window.addEventListener('load', () => {
	getSignStatus();
	checkDebugMode();
});
