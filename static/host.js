const socket = io();
const video = document.getElementById('host-video');
const status = document.getElementById('host-status');
const socketStatus = document.getElementById('socket-status');
const videoResolution = document.getElementById('video-resolution');
const videoFps = document.getElementById('video-fps');
const framesSent = document.getElementById('frames-sent');

let frameCount = 0;
let lastFpsUpdate = Date.now();

socket.on('connect', () => {
	socketStatus.textContent = 'Connected';
	socketStatus.style.color = '#4ade80';
});
socket.on('disconnect', () => {
	socketStatus.textContent = 'Disconnected';
	socketStatus.style.color = '#ef4444';
});

async function startCamera() {
	try {
		const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
		video.srcObject = stream;
		status.textContent = 'Streaming to server...';
		const track = stream.getVideoTracks()[0];
		const imageCapture = new ImageCapture(track);
		// Set resolution info
		video.onloadedmetadata = () => {
			videoResolution.textContent = `${video.videoWidth}x${video.videoHeight}`;
		};
		// Send frames at 30fps max
		let lastFrameTime = 0;
		async function sendFrame() {
			const now = performance.now();
			if (now - lastFrameTime < 33) {
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
			// Update FPS every second
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
	}
}
startCamera();
