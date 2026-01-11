import json
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split



def adjust_sequence_length(sequence, target_len=40):
    """
    Resamples a sequence of landmarks to a fixed length.
    Works for both shortening and lengthening.
    """
    current_len = len(sequence)
    
    # Create an array of indices for the current sequence
    # e.g., if current_len is 100 and target_len is 40: [0, 2.5, 5, ..., 99]
    indices = np.linspace(0, current_len - 1, target_len)
    
    # Grab the frames at those calculated indices
    # We use integer rounding to pick the closest real frame
    resampled_sequence = [sequence[int(i)] for i in indices]
    
    return np.array(resampled_sequence)





# --- STEP 1 & 2: Load Mapping ---
with open('WordModel/WLASL_v03.json', 'r') as f:
    mapping_data = json.load(f)

video_to_label = {}
for entry in mapping_data:
    gloss = entry['gloss']
    for inst in entry['instances']:
        video_to_label[inst['video_id']] = gloss

# --- STEP 3: Only include labels for files that EXIST ---
sequences, labels_strings = [], []
processed_dir = "WLASL-landmarks"

for video_id, gloss in video_to_label.items():
    npy_path = os.path.join(processed_dir, f"{video_id}.npy")
    if os.path.exists(npy_path):
        res = np.load(npy_path)
        # Verify if the sequence is empty or corrupted
        if res.shape[0] > 0:
            sequences.append(res)
            labels_strings.append(gloss)

# Now define actions based ONLY on what we actually loaded
actions = np.array(sorted(list(set(labels_strings))))
label_map = {label: num for num, label in enumerate(actions)}

# Convert string labels to integers
labels = [label_map[l] for l in labels_strings]

# --- STEP 4: Standardize Sequences ---
processed_sequences = [adjust_sequence_length(seq, target_len=40) for seq in sequences]

X = np.array(processed_sequences) 
y = to_categorical(labels).astype(int) 

# --- STEP 5: Model Input Shape Fix ---
# Note: Check if your data is 126 (both hands) or 63 (one hand)
input_feat_dim = X.shape[2] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(40, input_feat_dim)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax') # Dynamically set to 1994
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))


# Save the weights
model.save_weights('./WordModel/checkpoints/wordmodels')