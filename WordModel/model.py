import json
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. Load Mapping
with open('wlasl_v0.3.json', 'r') as f:
    mapping_data = json.load(f)

# 2. Create Lookup: {video_id: gloss}
video_to_label = {}
for entry in mapping_data:
    gloss = entry['gloss']
    for inst in entry['instances']:
        video_to_label[inst['video_id']] = gloss

# 3. Define your Classes (Top 100 or all)
actions = np.array(sorted(list(set(video_to_label.values()))))
label_map = {label:num for num, label in enumerate(actions)}


sequences, labels = [], []
processed_dir = "path/to/your/landmarks_folder"

for video_id, gloss in video_to_label.items():
    npy_path = os.path.join(processed_dir, f"{video_id}.npy")
    if os.path.exists(npy_path):
        res = np.load(npy_path)
        sequences.append(res)
        labels.append(label_map[gloss])

X = np.array(sequences) # Shape: (samples, 30, 126)
y = to_categorical(labels).astype(int) # One-hot encoding

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)