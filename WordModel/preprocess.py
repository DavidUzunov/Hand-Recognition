import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import json


# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Extract 21 landmarks (x, y, z) for one hand
        # Note: For simplicity, this grabs the first hand detected
        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            sequence.append(landmarks)
        else:
            # If no hand detected, use zeros as a placeholder
            sequence.append([0] * 63) 
            
    cap.release()
    return np.array(sequence)


def preprocess_video(video_path, sequence_length=30):
    cap = cv2.VideoCapture(video_path)
    window = []
    
    while len(window) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            # If video ends early, pad with zeros or last known frame
            while len(window) < sequence_length:
                window.append(np.zeros(126)) # 21 points * 3 coords * 2 hands
            break
            
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Extract coordinates for both hands
        frame_landmarks = []
        for hand_idx in range(2): # Process 2 hands
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > hand_idx:
                for lm in results.multi_hand_landmarks[hand_idx].landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                # Fill with zeros if hand not found
                frame_landmarks.extend([0] * 63)
                
        window.append(frame_landmarks)
        
    cap.release()
    return np.array(window)

# Example: Process a folder
# data = preprocess_video("my_video.mp4")
# np.save("processed_data/hello_1.npy", data)

def bulk_process_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use a for loop to process every video
    for video_file in input_path.glob("*.mp4"):
        # Determine where to save the output
        save_name = output_path / f"{video_file.stem}.npy"
        
        # Skip if already processed
        if save_name.exists():
            continue
            
        print(f"Extracting landmarks from {video_file.name}...")
        
        # Call your landmark extraction function (defined previously)
        landmarks = extract_landmarks(str(video_file))
        
        # Save as a binary file for training
        np.save(save_name, landmarks)

# Run the process
bulk_process_dataset("../Get-Data/WLASL/wlasl-complete/videos", "WLASL-landmarks")