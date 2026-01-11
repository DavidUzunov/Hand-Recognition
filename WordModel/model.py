"""
ASL Recognition Model Architecture

This module provides a TensorFlow/Keras model for recognizing ASL (American Sign Language)
gestures from video sequences. The model processes sequences of frames and outputs
predictions for different ASL words/signs.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json


class ASLRecognitionModel:
    """3D CNN model for ASL gesture recognition from video sequences."""
    
    def __init__(
        self,
        input_shape=(30, 224, 224, 3),
        num_classes=26,  # A-Z alphabet
        model_path="models/asl_model.h5",
        classes_path="models/classes.json",
    ):
        """
        Initialize the ASL Recognition model.
        
        Args:
            input_shape: Tuple of (sequence_length, height, width, channels)
            num_classes: Number of gesture classes to predict
            model_path: Path to save/load the model
            classes_path: Path to save/load class mappings
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_path = model_path
        self.classes_path = classes_path
        self.model = None
        self.class_names = None
        self.history = None
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    def build_model(self):
        """Build the 3D CNN model architecture for video sequence classification."""
        model = keras.Sequential([
            # Input layer: (seq_len, height, width, channels)
            layers.Input(shape=self.input_shape),
            
            # 3D Convolutional Blocks - extract spatiotemporal features
            # Block 1: 32 filters
            layers.Conv3D(
                32, 
                kernel_size=(3, 3, 3),
                padding="same",
                activation="relu"
            ),
            layers.BatchNormalization(),
            layers.Conv3D(
                32, 
                kernel_size=(3, 3, 3),
                padding="same",
                activation="relu"
            ),
            layers.BatchNormalization(),
            layers.MaxPool3D(pool_size=(2, 2, 2)),
            layers.Dropout(0.3),
            
            # Block 2: 64 filters
            layers.Conv3D(
                64, 
                kernel_size=(3, 3, 3),
                padding="same",
                activation="relu"
            ),
            layers.BatchNormalization(),
            layers.Conv3D(
                64, 
                kernel_size=(3, 3, 3),
                padding="same",
                activation="relu"
            ),
            layers.BatchNormalization(),
            layers.MaxPool3D(pool_size=(2, 2, 2)),
            layers.Dropout(0.3),
            
            # Block 3: 128 filters
            layers.Conv3D(
                128, 
                kernel_size=(3, 3, 3),
                padding="same",
                activation="relu"
            ),
            layers.BatchNormalization(),
            layers.Conv3D(
                128, 
                kernel_size=(3, 3, 3),
                padding="same",
                activation="relu"
            ),
            layers.BatchNormalization(),
            layers.MaxPool3D(pool_size=(2, 2, 2)),
            layers.Dropout(0.3),
            
            # Global Average Pooling - reduce spatial dimensions
            layers.GlobalAveragePooling3D(),
            
            # Dense layers - classification head
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer - softmax for multi-class classification
            layers.Dense(self.num_classes, activation="softmax"),
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=1e-3):
        """Compile the model with optimizer and loss function."""
        if self.model is None:
            self.build_model()
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")],
        )
        return self.model
    
    def train(
        self,
        train_generator,
        validation_generator,
        epochs=50,
        verbose=1,
        callbacks=None,
    ):
        """
        Train the model on gesture sequences.
        
        Args:
            train_generator: Generator yielding (X_batch, y_batch) for training
            validation_generator: Generator yielding (X_batch, y_batch) for validation
            epochs: Number of training epochs
            verbose: Verbosity level (0, 1, or 2)
            callbacks: List of Keras callbacks
            
        Returns:
            history: Training history object
        """
        if self.model is None:
            self.compile_model()
        
        # Default callbacks: model checkpointing and early stopping
        if callbacks is None:
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    self.model_path,
                    monitor="val_accuracy",
                    save_best_only=True,
                    mode="max",
                    verbose=1,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=10,
                    mode="max",
                    verbose=1,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1,
                ),
            ]
        
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )
        
        return self.history
    
    def set_class_names(self, class_names):
        """Set the class names for predictions."""
        if isinstance(class_names, dict):
            self.class_names = class_names
        else:
            # Assume list of class names
            self.class_names = {i: name for i, name in enumerate(class_names)}
    
    def predict(self, sequence, confidence_threshold=0.3):
        """
        Predict gesture class from a video sequence.
        
        Args:
            sequence: Numpy array of shape (seq_len, height, width, 3) or (1, seq_len, height, width, 3)
            confidence_threshold: Minimum confidence to return a prediction
            
        Returns:
            Dictionary with keys:
                - 'class_name': Predicted gesture class name
                - 'class_id': Predicted class index
                - 'confidence': Confidence score
                - 'all_probs': Dictionary of all class probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Handle input shape
        if len(sequence.shape) == 4:
            sequence = sequence[np.newaxis, ...]
        
        # Get predictions
        predictions = self.model.predict(sequence, verbose=0)
        probs = predictions[0]
        
        # Get top prediction
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        
        # Get class name if available
        class_name = self.class_names.get(class_id, f"class_{class_id}") if self.class_names else f"class_{class_id}"
        
        # Return None if below threshold
        if confidence < confidence_threshold:
            return {
                "class_name": None,
                "class_id": None,
                "confidence": None,
                "all_probs": {self.class_names.get(i, f"class_{i}"): float(p) for i, p in enumerate(probs)},
            }
        
        return {
            "class_name": class_name,
            "class_id": class_id,
            "confidence": confidence,
            "all_probs": {self.class_names.get(i, f"class_{i}"): float(p) for i, p in enumerate(probs)},
        }
    
    def save_model(self):
        """Save the model to disk."""
        if self.model is not None:
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")
        
        # Save class names mapping
        if self.class_names is not None:
            with open(self.classes_path, "w") as f:
                # Convert to string keys for JSON compatibility
                class_dict = {str(k): v for k, v in self.class_names.items()}
                json.dump(class_dict, f)
            print(f"Class names saved to {self.classes_path}")
    
    def load_model(self):
        """Load a trained model from disk."""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load class names if available
        if os.path.exists(self.classes_path):
            with open(self.classes_path, "r") as f:
                class_dict = json.load(f)
                # Convert string keys back to integers
                self.class_names = {int(k): v for k, v in class_dict.items()}
            print(f"Class names loaded from {self.classes_path}")
        
        return self.model
    
    def summary(self):
        """Print model architecture summary."""
        if self.model is None:
            self.build_model()
        self.model.summary()
    
    def get_config(self):
        """Get model configuration for debugging/logging."""
        return {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "model_path": self.model_path,
            "classes_path": self.classes_path,
            "num_parameters": self.model.count_params() if self.model else None,
        }


# Import numpy if needed for the module
import numpy as np


if __name__ == "__main__":
    # Quick test of model architecture
    model_config = ASLRecognitionModel(
        input_shape=(30, 224, 224, 3),
        num_classes=26,  # A-Z
    )
    model_config.build_model()
    model_config.compile_model()
    model_config.summary()
    print("\nModel Config:")
    print(model_config.get_config())
