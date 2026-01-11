import cv2
import mediapipe as mp
import numpy as np
import json
import os

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

# Example usage with MS-ASL JSON
# with open('MSASL_train.json') as f:
#    data = json.load(f)
#    # Loop through data, extract landmarks from 'file' path, and save as .npy