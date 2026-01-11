// Host page logic
const socket = io();
const video = document.getElementById('host-video');
const status = document.getElementById('host-status');
const socketStatus = document.getElementById('socket-status');
const videoResolution = document.getElementById('video-resolution');
const videoFps = document.getElementById('video-fps');
const framesSent = document.getElementById('frames-sent');
const cameraSelect = document.getElementById('camera-select');
const startBtn = document.getElementById('start-camera-btn');

// Add Stop Camera button
let stopBtn = document.getElementById('stop-camera-btn');
if (!stopBtn) {
	stopBtn = document.createElement('button');
	stopBtn.id = 'stop-camera-btn';
	stopBtn.className = 'admin-btn';
	stopBtn.textContent = 'Stop Camera';
	stopBtn.style.marginLeft = '10px';
	stopBtn.style.display = 'none';
	document.getElementById('camera-select-section').appendChild(stopBtn);
}

const previewVideo = document.getElementById('camera-preview');
let selectedDeviceId = null;
let stream = null;
let streaming = false;
let frameCount = 0;
let lastFpsUpdate = Date.now();
let stopStreamFn = null;
let previewStream = null;

socket.on('connect', () => {
	socketStatus.textContent = 'Connected';
	socketStatus.style.color = '#4ade80';
});
socket.on('disconnect', () => {
	socketStatus.textContent = 'Disconnected';
	socketStatus.style.color = '#ef4444';
});

async function populateCameras() {
	try {
		const devices = await navigator.mediaDevices.enumerateDevices();
		const videoDevices = devices.filter(d => d.kind === 'videoinput');
		cameraSelect.innerHTML = '';
		videoDevices.forEach((device, idx) => {
			const option = document.createElement('option');
			option.value = device.deviceId;
			option.textContent = device.label || `Camera ${idx + 1}`;
			cameraSelect.appendChild(option);
		});
		if (videoDevices.length > 0) {
			selectedDeviceId = videoDevices[0].deviceId;
		}
	} catch (e) {
		status.textContent = 'Could not enumerate cameras.';
	}
}

async function showCameraPreview(deviceId) {
	if (previewStream) {
		previewStream.getTracks().forEach(track => track.stop());
		previewStream = null;
	}
	if (!deviceId) {
		previewVideo.style.display = 'none';
		return;
	}
	try {
		const constraints = { video: { deviceId: { exact: deviceId } }, audio: false };
		previewStream = await navigator.mediaDevices.getUserMedia(constraints);
		previewVideo.srcObject = previewStream;
		previewVideo.style.display = '';
	} catch (e) {
		previewVideo.style.display = 'none';
	}
}

cameraSelect.addEventListener('change', e => {
	if (!streaming) {
		selectedDeviceId = cameraSelect.value;
		showCameraPreview(selectedDeviceId);
	}
});

startBtn.addEventListener('click', async () => {
	// Always get the current value from the dropdown
	selectedDeviceId = cameraSelect.value;
	if (!selectedDeviceId) {
		status.textContent = 'Please select a camera.';
		return;
	}
	cameraSelect.disabled = true;
	startBtn.disabled = true;
	stopBtn.style.display = '';
	document.getElementById('camera-select-section').style.display = 'none';
	video.style.display = '';
	status.textContent = 'Initializing camera...';
	if (previewStream) {
		previewStream.getTracks().forEach(track => track.stop());
		previewStream = null;
	}
	previewVideo.style.display = 'none';
	await startHostCamera(selectedDeviceId);
});

stopBtn.addEventListener('click', () => {
	if (stopStreamFn) stopStreamFn();
	if (stream) {
		stream.getTracks().forEach(track => track.stop());
		stream = null;
	}
	streaming = false;
	video.srcObject = null;
	video.style.display = 'none';
	stopBtn.style.display = 'none';
	cameraSelect.disabled = false;
	startBtn.disabled = false;
	document.getElementById('camera-select-section').style.display = '';
	status.textContent = 'Camera stopped. Select a camera to start.';
	videoResolution.textContent = '-';
	videoFps.textContent = '-';
	framesSent.textContent = '0';
});

window.startHostCamera = async function (deviceId) {
	try {
		const constraints = { video: { deviceId: { exact: deviceId } }, audio: false };
		stream = await navigator.mediaDevices.getUserMedia(constraints);
		video.srcObject = stream;
		status.textContent = 'Streaming to server...';
		const track = stream.getVideoTracks()[0];
		const imageCapture = new ImageCapture(track);
		streaming = true;
		video.onloadedmetadata = () => {
			videoResolution.textContent = `${video.videoWidth}x${video.videoHeight}`;
		};
		let lastFrameTime = 0;
		let running = true;
		stopStreamFn = () => { running = false; };
		async function sendFrame() {
			if (!running) return;
			const now = performance.now();
			if (now - lastFrameTime < 67) {
				requestAnimationFrame(sendFrame);
				return;
			}
			lastFrameTime = now;
			try {
				const bitmap = await imageCapture.grabFrame();
				const canvas = document.createElement('canvas');
				canvas.width = bitmap.width;
				canvas.height = bitmap.height;
				const ctx = canvas.getContext('2d');
				ctx.drawImage(bitmap, 0, 0);
				canvas.toBlob(async blob => {
					if (blob) {
						try {
							const buffer = await blob.arrayBuffer();
							socket.emit('host_frame', new Uint8Array(buffer));
							frameCount++;
							framesSent.textContent = frameCount;
						} catch (e) {
							console.error('Error converting blob to arrayBuffer:', e);
						}
					}
				}, 'image/jpeg', 0.7);
			} catch (e) {
				console.error('Error grabbing frame:', e);
			}
			const now2 = Date.now();
			if (now2 - lastFpsUpdate > 1000) {
				videoFps.textContent = frameCount;
				frameCount = 0;
				lastFpsUpdate = now2;
			}
			requestAnimationFrame(sendFrame);
		}
		sendFrame();
	} catch (err) {
		status.textContent = 'Camera access denied or unavailable.';
		streaming = false;
		stopBtn.style.display = 'none';
		cameraSelect.disabled = false;
		startBtn.disabled = false;
	}
}

// Initial camera list
populateCameras().then(() => {
	if (selectedDeviceId) showCameraPreview(selectedDeviceId);
});
