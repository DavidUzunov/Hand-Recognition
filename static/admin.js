// Import common logic
// import './common.js';

// Admin-specific JS
// Listen for camera_status events and update status
const adminSocket = io();
adminSocket.on('camera_status', (data) => {
	if (window.updateCameraStatus) window.updateCameraStatus(data);
});

// Admin page logic here.
