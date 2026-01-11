// Common JS for all pages
// Add shared socket logic, utility functions, and UI helpers here.

// Example: Utility to update camera status
function updateCameraStatus(data) {
	const cameraHeader = document.getElementById('camera-status-header');
	const selectedCamera = document.getElementById('selected-camera');
	if (cameraHeader) {
		cameraHeader.textContent = data.camera_available ? 'Available' : 'Unavailable';
	}
	if (selectedCamera) {
		selectedCamera.textContent = data.device || '-';
	}
}

// Export for use in other scripts
window.updateCameraStatus = updateCameraStatus;
