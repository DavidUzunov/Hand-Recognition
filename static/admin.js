// Import common logic
// import './common.js';

// Admin-specific JS
const adminSocket = io();
const toggleBtn = document.getElementById('toggle-asl-btn');
const cameraSelect = document.getElementById('camera_id');
const transcribeMode = document.getElementById('transcribe-mode');
const socketStatus = document.getElementById('socket-status');
const pingTime = document.getElementById('ping-time');
let lastPing = null;

adminSocket.on('connect', function () {
	if (socketStatus) socketStatus.textContent = 'Connected';
});
adminSocket.on('disconnect', function () {
	if (!socketStatus) socketStatus.textContent = 'Disconnected';
});
adminSocket.on('ping', function (data) {
	if (pingTime && data.timestamp) {
		const now = Date.now() / 1000;
		const latency = Math.round((now - data.timestamp) * 1000);
		pingTime.textContent = latency + ' ms';
		lastPing = now;
	}
});

function setTranscriptionControlsDisabled(disabled) {
	if (cameraSelect) cameraSelect.disabled = disabled;
	if (transcribeMode) transcribeMode.disabled = disabled;
	const changeBtn = document.querySelector('.camera-selector button[type="submit"]');
	if (changeBtn) changeBtn.disabled = disabled;
}

adminSocket.on('sign_status', function (data) {
	if (toggleBtn) {
		toggleBtn.textContent = data.signing_active ? 'Stop ASL Transcription' : 'Start ASL Transcription';
	}
	setTranscriptionControlsDisabled(!!data.signing_active);
});
toggleBtn.addEventListener('click', function () {
	adminSocket.emit('toggle_sign');
});
adminSocket.emit('get_sign_status');

// Listen for camera_status events and update status
adminSocket.on('camera_status', (data) => {
	if (window.updateCameraStatus) window.updateCameraStatus(data);
});

// Admin page logic here.
