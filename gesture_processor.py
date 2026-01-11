"""
Thread-safe gesture processor for ASL sign language transcription.
Accumulates hand gesture frames and transcribes them to text.
"""

import threading
import time
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
import tensorflow as tf


@dataclass
class GestureFrame:
    """A single frame with hand landmark data"""
    timestamp: float
    hand_landmarks: object  # mediapipe hand landmarks
    frame_id: int


class ThreadSafeGestureProcessor:
    """
    Thread-safe processor for ASL gestures.
    
    - Receives hand landmark frames from the main thread
    - Predicts letters using the ASL model
    - Accumulates letters into words using gesture duration and silence detection
    - Calls a callback when words are complete
    """
    
    LETTERS = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    ]
    
    def __init__(self, model_path: str):
        """Initialize processor with ASL model"""
        self.lock = threading.RLock()
        self.should_stop = False
        
        # Load model with thread-safe protection
        with self.lock:
            self.model = tf.keras.models.load_model(model_path)
        
        # Gesture and word state
        self.gesture_queue = Queue(maxsize=50)
        self.current_word = ""
        self.gesture_buffer = []  # [(letter, start_time, end_time), ...]
        self.last_letter = None
        self.last_gesture_time = 0
        
        # Configuration
        self.silence_threshold = 0.5  # seconds of no detection to end word
        self.gesture_duration_threshold = 0.3  # seconds to detect double letter
        
        # Callback
        self.transcript_callback: Optional[Callable] = None
        
        print("GestureProcessor initialized with model:", model_path)
    
    def set_callback(self, callback: Callable[[str], None]):
        """Set callback for completed word transcriptions"""
        with self.lock:
            self.transcript_callback = callback
    
    def add_gesture(self, frame: GestureFrame):
        """
        Add a gesture frame to processing queue (thread-safe).
        Called from web.py socket handler.
        """
        try:
            self.gesture_queue.put_nowait(frame)
        except:
            # Queue full, drop frame silently
            pass
    
    def _process_landmarks(self, frame: GestureFrame) -> Optional[str]:
        """
        Process hand landmarks and predict letter.
        Returns the predicted letter or None.
        """
        try:
            hand = frame.hand_landmarks
            
            # Normalize data via wrist (landmark 0)
            data_list = []
            for measurement in hand.landmark:
                data_list.append(measurement.x - hand.landmark[0].x)
                data_list.append(measurement.y - hand.landmark[0].y)
                data_list.append(measurement.z - hand.landmark[0].z)
            
            data = np.array(data_list).reshape(1, -1).astype(np.float32)
            
            # Predict with thread-safe model inference
            with self.lock:
                prediction = self.model.predict(data, verbose=0)
            
            letter_id = np.argmax(prediction)
            confidence = float(prediction[0][letter_id])
            letter = self.LETTERS[letter_id] if letter_id < len(self.LETTERS) else "?"
            
            return letter if confidence > 0.5 else None
            
        except Exception as e:
            print(f"Error processing hand landmarks: {e}")
            return None
    
    def _check_double_letter(self, letter: str, start_time: float, end_time: float) -> bool:
        """Check if a letter was held long enough to be considered a double"""
        duration = end_time - start_time
        return duration >= self.gesture_duration_threshold
    
    def _check_word_boundary(self, now: float):
        """
        Check if enough silence has passed to end the current word.
        Called periodically to detect word boundaries.
        """
        with self.lock:
            # If no gesture detected for silence_threshold seconds, end word
            if (self.last_gesture_time > 0 and 
                now - self.last_gesture_time >= self.silence_threshold and
                self.current_word):
                
                # Send completed word to callback
                if self.transcript_callback:
                    try:
                        self.transcript_callback(self.current_word)
                    except Exception as e:
                        print(f"Error calling transcript callback: {e}")
                
                # Reset for next word
                self.current_word = ""
                self.gesture_buffer = []
                self.last_letter = None
                self.last_gesture_time = 0
    
    def process(self):
        """
        Main processing loop (runs in dedicated thread).
        Reads gesture frames from queue and accumulates them into words.
        """
        print("GestureProcessor loop started")
        
        last_letter = None
        letter_start_time = None
        
        while not self.should_stop:
            try:
                # Get next frame with timeout to allow word boundary checks
                frame = self.gesture_queue.get(timeout=1.0)
                now = frame.timestamp
                
                # Predict letter from hand landmarks
                letter = self._process_landmarks(frame)
                
                if letter is None:
                    # No valid hand detected - might be start of silence
                    with self.lock:
                        self.last_gesture_time = now
                    continue
                
                # Track gesture duration for double letters
                if letter != last_letter:
                    # New letter detected
                    if last_letter is not None and letter_start_time is not None:
                        # Save duration of previous letter
                        duration = now - letter_start_time
                        is_double = duration >= self.gesture_duration_threshold
                        
                        with self.lock:
                            self.current_word += last_letter
                            if is_double:
                                self.current_word += last_letter
                    
                    # Start tracking new letter
                    last_letter = letter
                    letter_start_time = now
                    
                    with self.lock:
                        self.last_gesture_time = now
                
            except Empty:
                # Timeout - check for word boundaries
                if not self.should_stop:
                    now = time.time()
                    self._check_word_boundary(now)
            
            except Exception as e:
                if not self.should_stop:
                    print(f"Error in gesture processor loop: {e}")
        
        print("GestureProcessor loop stopped")
    
    def get_transcript(self) -> str:
        """Get current accumulated word (thread-safe)"""
        with self.lock:
            return self.current_word
    
    def reset(self):
        """Reset processor state (thread-safe)"""
        with self.lock:
            self.current_word = ""
            self.gesture_buffer = []
            self.last_letter = None
            self.last_gesture_time = 0
    
    def flush_word(self):
        """Force the current word to be sent (thread-safe)"""
        with self.lock:
            if self.current_word and self.transcript_callback:
                try:
                    self.transcript_callback(self.current_word)
                except Exception as e:
                    print(f"Error calling transcript callback: {e}")
            self.current_word = ""
            self.gesture_buffer = []
            self.last_letter = None
            self.last_gesture_time = 0
    
    def stop(self):
        """Stop the processor gracefully"""
        with self.lock:
            self.should_stop = True
        print("GestureProcessor stop signal sent")
