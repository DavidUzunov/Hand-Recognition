#!/usr/bin/env python3
"""
ASL Recognition Model Training Script

This script trains the ASL recognition model on labeled gesture data.
Data should be organized as follows:

data/
├── train/
│   ├── A/
│   │   ├── gesture_1.npy     (shape: (seq_len, 224, 224, 3), dtype: float32, normalized [0,1])
│   │   ├── gesture_2.npy
│   │   └── ...
│   ├── B/
│   │   ├── gesture_1.npy
│   │   └── ...
│   └── ...
├── val/
│   ├── A/
│   ├── B/
│   └── ...
└── test/
    ├── A/
    ├── B/
    └── ...

Each .npy file should contain a sequence of preprocessed frames:
- Shape: (sequence_length, 224, 224, 3)
- dtype: float32
- Values: normalized to [0, 1] range
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path
import argparse
from sklearn.utils import shuffle

# Import the model class
from model import ASLRecognitionModel


class GestureDataGenerator(keras.utils.Sequence):
    """Custom data generator for loading gesture sequences from .npy files."""
    
    def __init__(
        self,
        data_dir,
        batch_size=16,
        seq_len=30,
        target_size=(224, 224),
        shuffle_data=True,
    ):
        """
        Initialize the data generator.
        
        Args:
            data_dir: Path to data directory (train, val, or test)
            batch_size: Number of samples per batch
            seq_len: Expected sequence length (will pad/truncate to this)
            target_size: Target spatial dimensions for each frame (height, width)
            shuffle_data: Whether to shuffle data after each epoch
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.target_size = target_size
        self.shuffle_data = shuffle_data
        
        # Load file paths and labels
        self.file_paths = []
        self.labels = []
        self.class_names = {}
        self.class_id_map = {}
        
        self._load_data()
        
        if shuffle_data:
            self._shuffle()
    
    def _load_data(self):
        """Load all gesture files and their labels."""
        class_id = 0
        
        # Iterate through subdirectories (each is a class)
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            self.class_names[class_id] = class_name
            self.class_id_map[class_name] = class_id
            
            # Load all .npy files in this class directory
            for npy_file in sorted(class_dir.glob("*.npy")):
                self.file_paths.append(str(npy_file))
                self.labels.append(class_id)
            
            class_id += 1
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No .npy files found in {self.data_dir}")
        
        print(f"Loaded {len(self.file_paths)} samples from {len(self.class_names)} classes")
        for cid, cname in sorted(self.class_names.items()):
            count = sum(1 for l in self.labels if l == cid)
            print(f"  {cname}: {count} samples")
    
    def _shuffle(self):
        """Shuffle the data."""
        indices = np.random.permutation(len(self.file_paths))
        self.file_paths = [self.file_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
    
    def __len__(self):
        """Return number of batches per epoch."""
        return int(np.ceil(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """Load and return a batch of data."""
        batch_files = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        
        X = []
        y = []
        
        for file_path, label in zip(batch_files, batch_labels):
            try:
                # Load the sequence
                seq = np.load(file_path)
                
                # Validate shape
                if seq.ndim != 4 or seq.shape[1:] != (self.target_size[0], self.target_size[1], 3):
                    print(f"Warning: {file_path} has unexpected shape {seq.shape}, skipping")
                    continue
                
                # Pad or truncate to seq_len
                if seq.shape[0] < self.seq_len:
                    # Pad with copies of the last frame
                    pad_count = self.seq_len - seq.shape[0]
                    last_frame = seq[-1:].repeat(pad_count, axis=0)
                    seq = np.concatenate([seq, last_frame], axis=0)
                elif seq.shape[0] > self.seq_len:
                    # Take first seq_len frames
                    seq = seq[:self.seq_len]
                
                X.append(seq)
                y.append(label)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # One-hot encode labels
        num_classes = len(self.class_names)
        y = keras.utils.to_categorical(y, num_classes=num_classes)
        
        return X, y
    
    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if self.shuffle_data:
            self._shuffle()


def create_dummy_data(data_dir="data", num_classes=26):
    """
    Create dummy gesture data for testing.
    
    Creates random sequences in the expected format.
    """
    data_path = Path(data_dir)
    class_names = [chr(65 + i) for i in range(num_classes)]  # A-Z
    
    for split in ["train", "val"]:
        split_dir = data_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 10 samples per class for training, 3 for validation
        num_samples = 10 if split == "train" else 3
        
        for class_name in class_names:
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_samples):
                # Create random sequence data
                seq = np.random.uniform(0, 1, size=(30, 224, 224, 3)).astype(np.float32)
                
                # Save as .npy file
                npy_path = class_dir / f"gesture_{i:03d}.npy"
                np.save(npy_path, seq)
        
        print(f"Created dummy {split} data in {split_dir}")


def train_model(
    data_dir="data",
    model_name="asl_model.h5",
    epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    seq_len=30,
    create_dummy=False,
):
    """
    Train the ASL recognition model.
    
    Args:
        data_dir: Path to data directory containing train/val subdirectories
        model_name: Name of the model file to save
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        seq_len: Sequence length for video clips
        create_dummy: Whether to create dummy data for testing
    """
    # Create dummy data if requested
    if create_dummy:
        print("Creating dummy data for testing...")
        create_dummy_data(data_dir)
    
    # Check if data directories exist
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        raise ValueError(
            f"Data directories not found. Expected:\n"
            f"  {train_dir}\n"
            f"  {val_dir}\n"
            f"\nRun with --create-dummy to create test data."
        )
    
    # Create data generators
    print("\nLoading training data...")
    train_gen = GestureDataGenerator(
        str(train_dir),
        batch_size=batch_size,
        seq_len=seq_len,
        shuffle_data=True,
    )
    
    print("\nLoading validation data...")
    val_gen = GestureDataGenerator(
        str(val_dir),
        batch_size=batch_size,
        seq_len=seq_len,
        shuffle_data=False,
    )
    
    # Create and compile model
    print("\nBuilding model...")
    num_classes = len(train_gen.class_names)
    model_path = f"models/{model_name}"
    classes_path = f"models/classes.json"
    
    asl_model = ASLRecognitionModel(
        input_shape=(seq_len, 224, 224, 3),
        num_classes=num_classes,
        model_path=model_path,
        classes_path=classes_path,
    )
    
    asl_model.set_class_names(train_gen.class_names)
    asl_model.build_model()
    asl_model.compile_model(learning_rate=learning_rate)
    asl_model.summary()
    
    # Train model
    print(f"\nTraining model for {epochs} epochs...")
    print(f"Model: {model_path}")
    print(f"Classes: {classes_path}")
    
    history = asl_model.train(
        train_gen,
        val_gen,
        epochs=epochs,
        verbose=1,
    )
    
    # Save model
    print("\nSaving model...")
    asl_model.save_model()
    
    # Plot training history if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        axes[0].plot(history.history["accuracy"], label="Train Accuracy")
        axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Model Accuracy")
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(history.history["loss"], label="Train Loss")
        axes[1].plot(history.history["val_loss"], label="Val Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Model Loss")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig("models/training_history.png", dpi=150)
        print("Training history saved to models/training_history.png")
        
    except ImportError:
        print("Matplotlib not available; skipping training history plot")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL recognition model")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--model-name",
        default="asl_model.h5",
        help="Model file name (default: asl_model.h5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=30,
        help="Sequence length (default: 30)",
    )
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="Create dummy data for testing",
    )
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seq_len=args.seq_len,
        create_dummy=args.create_dummy,
    )
