# Hand Recognition - ASL Neural Network Pipeline

This project implements a complete machine learning pipeline for American Sign Language (ASL) recognition using deep learning and video analysis.

## Quick Start

### 1. Install Dependencies

```bash
# If not already installed
python bootstrap.py
```

Or manually:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Training Data

You can prepare data from video files or image sequences:

#### From Video Files
```bash
python prepare_data.py \
    --input-videos /path/to/gesture/videos \
    --output-dir ./data \
    --num-frames 30
```

Expected directory structure:
```
/path/to/gesture/videos/
├── A/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── B/
│   └── ...
└── ...
```

#### From Image Sequences
```bash
python prepare_data.py \
    --input-images /path/to/sequences \
    --output-dir ./data \
    --num-frames 30
```

Expected structure:
```
/path/to/sequences/
├── A_seq1/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
├── A_seq2/
│   └── ...
└── ...
```

#### Using Dummy Data (for testing)
```bash
python train.py --create-dummy --epochs 5
```

### 3. Train the Model

```bash
python train.py \
    --data-dir ./data \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 1e-3
```

The trained model will be saved in:
- `models/asl_model.h5` - Model weights
- `models/classes.json` - Class name mappings
- `models/training_history.png` - Training curves

### 4. Run the Application

```bash
python bootstrap.py
```

The server will start on `http://0.0.0.0:5000`

## Architecture

### Model Architecture

The ASL recognition model is a **3D Convolutional Neural Network (3D CNN)**:

```
Input: (30, 224, 224, 3)  [seq_len, height, width, channels]
    ↓
3D Conv Block 1: 32 filters
    ↓
3D Conv Block 2: 64 filters
    ↓
3D Conv Block 3: 128 filters
    ↓
Global Average Pooling
    ↓
Dense Layers: 256 → 128
    ↓
Output: (num_classes,) [softmax]
```

**Key Features:**
- 3D convolutions capture spatiotemporal features from video sequences
- Batch normalization and dropout for regularization
- Global average pooling reduces parameters and prevents overfitting
- Early stopping and model checkpointing during training

### Data Format

Each gesture sample is stored as a `.npy` file:
- **Shape**: `(sequence_length, 224, 224, 3)`
- **Data Type**: `float32`
- **Value Range**: `[0.0, 1.0]` (normalized RGB)
- **Order**: RGB (not BGR)

### Pipeline Flow

```
Video Input (30 fps)
    ↓
Frame Capture & Buffering (30 frames)
    ↓
Preprocessing (224×224, RGB, normalize)
    ↓
3D CNN Model
    ↓
Softmax Classification
    ↓
Gesture Prediction + Confidence
    ↓
WebSocket → Frontend Display
```

## Usage

### Web Interface

1. Open browser to `http://localhost:5000`
2. Click "Start Sign Capture" button to begin gesture recording
3. Perform ASL sign/gesture in front of camera
4. The model will predict the gesture and display:
   - **Predicted Class**: Letter/sign recognized
   - **Confidence Score**: 0-100% confidence
   - **Probabilities**: All class predictions

### API Endpoints

#### GET /camera_status
Get current camera information:
```json
{
  "camera_id": 0,
  "camera_available": true,
  "device": "/dev/video0",
  "available_cameras": [0]
}
```

#### GET /video_feed
Stream MJPEG video with hand landmarks (if available).

#### WebSocket Events

**toggle_sign** - Start/stop gesture capture:
```python
socket.emit('toggle_sign', {'active': True})
```

**asl_transcript** - Receive gesture predictions:
```python
socket.on('asl_transcript', function(data) {
  console.log(data);
  // {
  //   "type": "gesture_prediction",
  //   "class_name": "A",
  //   "class_id": 0,
  //   "confidence": 0.95,
  //   "all_probs": {"A": 0.95, "B": 0.03, ...}
  // }
});
```

## File Structure

```
Hand-Recognition/
├── app.py                 # Backend core (camera, frame buffer, inference)
├── web.py                 # Flask/WebSocket server
├── bootstrap.py           # Entry point with auto-relaunch
├── model.py              # ASL recognition model class
├── train.py              # Training script
├── prepare_data.py       # Data preparation utility
├── requirements.txt      # Python dependencies
├── data/                 # Training data directory
│   ├── train/
│   ├── val/
│   └── test/
├── models/               # Trained model checkpoints
│   ├── asl_model.h5     # Trained weights
│   ├── classes.json     # Class mappings
│   └── training_history.png
├── static/               # Frontend assets
│   ├── app.js
│   └── style.css
├── templates/            # HTML templates
│   ├── index.html
│   └── 404.html
└── .venv/               # Python virtual environment
```

## Advanced Training Options

### Custom Learning Rate
```bash
python train.py --learning-rate 5e-4
```

### Different Batch Sizes
```bash
python train.py --batch-size 32
```

### Custom Model Name
```bash
python train.py --model-name gesture_recognition.h5
```

### Full Training Command
```bash
python train.py \
    --data-dir ./data \
    --model-name asl_model.h5 \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 1e-3 \
    --seq-len 30
```

## Model Predictions

The model outputs:
1. **class_name**: Predicted gesture/letter
2. **class_id**: Numeric class index
3. **confidence**: Prediction confidence (0-1)
4. **all_probs**: Dictionary of all class probabilities

### Confidence Threshold
Default threshold is 0.3 (30%). Predictions below this are filtered out.

Modify in `app.py`:
```python
prediction = predict_gesture(seq, confidence_threshold=0.5)
```

## Training Details

### Data Splits
- **Training**: 70% (used for learning)
- **Validation**: 15% (used for hyperparameter tuning)
- **Test**: 15% (reserved for final evaluation)

### Callbacks During Training
1. **ModelCheckpoint**: Saves best model based on validation accuracy
2. **EarlyStopping**: Stops if no improvement for 10 epochs
3. **ReduceLROnPlateau**: Reduces learning rate if validation loss plateaus

### Hardware Requirements
- **Minimum**: 4GB RAM, dual-core CPU
- **Recommended**: 8GB+ RAM, GPU (NVIDIA CUDA or Apple Silicon)
- **Training Time**: 10-30 minutes for 50 epochs (CPU), 2-5 minutes (GPU)

## Troubleshooting

### Model Not Loading
If `models/asl_model.h5` doesn't exist:
1. Train the model first: `python train.py --create-dummy`
2. Or use `prepare_data.py` to convert your own data

### Low Prediction Accuracy
1. Check data quality (clear lighting, consistent gestures)
2. Ensure at least 10 samples per class
3. Try more training epochs or higher learning rate
4. Collect more diverse samples

### WebSocket Connection Issues
1. Check browser console for connection errors
2. Ensure Flask-SocketIO is installed: `pip install Flask-SocketIO`
3. Try clearing browser cache

### Camera Not Detected
1. Check available cameras: `ls /dev/video*`
2. Verify camera permissions: `ls -l /dev/video*`
3. Try manually selecting camera: `curl http://localhost:5000/set_camera?id=1`

## Performance Notes

- **Inference Speed**: ~100-300ms per frame on CPU, ~10-50ms on GPU
- **Frame Buffer**: Holds 30 most recent frames
- **Sequence Length**: Default 30 frames (~1 second at 30 fps)
- **Model Size**: ~5-10MB on disk

## Extending the System

### Adding More Classes
1. Create subdirectories in `data/train/` for new classes
2. Add data samples with `prepare_data.py`
3. Retrain: `python train.py`

### Custom Post-Processing
Modify the callback in `web.py` to add custom logic:
```python
def send_asl_transcript(result):
    # Custom processing here
    pass
```

### Fine-tuning Pre-trained Models
Load and continue training:
```python
model = ASLRecognitionModel()
model.load_model()
model.train(train_gen, val_gen, epochs=10)
```

## References

- **3D CNN**: Tran et al. "Learning Spatiotemporal Features with 3D Convolutional Networks"
- **TensorFlow/Keras**: https://www.tensorflow.org/
- **MediaPipe**: https://mediapipe.dev/
- **OpenCV**: https://opencv.org/

## License

See LICENSE file in repository.
