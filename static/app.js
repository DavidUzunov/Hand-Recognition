const socket = io();

socket.on('connect', () => {
	console.log('Connected to server');
});

socket.on('ping', (data) => {
	console.log('Received ping from server:', data.message);
});

socket.on('camera_status', (data) => {
	console.log('Received camera status:', data);
	updateCameraStatus(data);
});

socket.on('sign_status', (data) => {
	console.log('Received sign status:', data);
	updateSignButton(data.signing_active);
});

socket.on('asl_transcript', (data) => {
	console.log('Received ASL transcript:', data.message);
	const messageBox = document.getElementById('message-box');
	if (messageBox) {
		const timestamp = new Date().toLocaleTimeString();
		messageBox.value += `[${timestamp}] ${data.message}\n`;
		// Auto-scroll to bottom
		messageBox.scrollTop = messageBox.scrollHeight;
	}
});

function updateCameraStatus(data) {
	try {
		const videoFeed = document.getElementById('video-feed');

		// Load video feed when camera is available, otherwise show placeholder
		if (data.camera_available) {
			videoFeed.src = '/video_feed';
		} else {
			videoFeed.src = '/no_camera';
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
	
	if (isActive) {
		button.textContent = 'Stop Signing';
		button.classList.add('active');
		if (sidebar) sidebar.style.display = 'flex';
	} else {
		button.textContent = 'Start Signing';
		button.classList.remove('active');
		if (sidebar) sidebar.style.display = 'none';
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
