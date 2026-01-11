# ASL Recognition Data Folder

This folder contains the training, validation, and test data for the ASL (American Sign Language) recognition model.

## Directory Structure

```
data/
├── train/          # Training data (70% of data)
│   ├── A/          # Letter A samples
│   │   ├── gesture_001.npy
│   │   ├── gesture_002.npy
│   │   └── ...
│   ├── B/          # Letter B samples
│   │   ├── gesture_001.npy
│   │   └── ...
│   └── ...
├── val/            # Validation data (15% of data)
│   ├── A/
│   ├── B/
│   └── ...
└── test/           # Test data (15% of data)
    ├── A/
    ├── B/
    └── ...
```

## File Format

Each gesture sample is stored as a `.npy` (NumPy) file containing:
- **Shape**: `(sequence_length, 224, 224, 3)`
- **Data Type**: `float32`
- **Value Range**: `[0.0, 1.0]` (normalized)
- **Order**: RGB (not BGR)

### Example

```python
import numpy as np

# Load a gesture sample
gesture = np.load('data/train/A/gesture_001.npy')
print(gesture.shape)  # (30, 224, 224, 3) for 30-frame sequence
print(gesture.dtype)  # float32
print(gesture.min(), gesture.max())  # Around 0.0 and 1.0
```

## Data Preparation Guide

### Method 1: From Video Files (Recommended)

Use the provided `prepare_data.py` script to convert video files into gesture sequences:

```bash
python prepare_data.py \
    --input-videos /path/to/gesture/videos \
    --output-dir ./data \
    --sequence-length 30 \
    --train-split 0.7 \
    --val-split 0.15
```

### Method 2: From Image Sequences

If you have image sequences (frames extracted from videos):

```bash
python prepare_data.py \
    --input-frames /path/to/frame/folders \
    --output-dir ./data \
    --sequence-length 30
```

### Method 3: Manual Creation

Create gesture samples programmatically:

```python
import numpy as np
from pathlib import Path

# Create a gesture sample
sequence = np.random.uniform(0, 1, (30, 224, 224, 3)).astype(np.float32)

# Save it
output_path = Path('data/train/A')
output_path.mkdir(parents=True, exist_ok=True)
np.save(output_path / 'gesture_001.npy', sequence)
```

## Class Organization

Currently supports **26 classes** (A-Z alphabet letters). You can modify the training script to support more classes:

- Modify the `num_classes` parameter in `train.py`
- Create subdirectories with appropriate class names
- Update class name mappings in `models/classes.json` after training

## Data Collection Tips

1. **Consistency**: Ensure uniform lighting and camera positioning across all recordings
2. **Variety**: Collect samples from different people to increase generalization
3. **Frame Extraction**: Extract 20-40 frames per gesture (model pads/truncates to 30)
4. **Resolution**: Maintain at least 224x224 resolution for each frame
5. **Quality**: Use high-quality cameras or webcams for better recognition accuracy

## Creating Test Data

To generate dummy data for testing:

```bash
python train.py --create-dummy --epochs 5
```

This creates random gesture samples in the correct format.

## Training

Once data is prepared, train the model:

```bash
python train.py \
    --data-dir ./data \
    --epochs 50 \
    --batch-size 16
```

The trained model and class mappings will be saved in `models/`:
- `models/asl_model.h5` - Trained model weights
- `models/classes.json` - Class name mappings
- `models/training_history.png` - Training curves (if matplotlib available)
