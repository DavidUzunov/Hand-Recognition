from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Suppose you have your labels in a list like ['hello', 'hello', 'thanks'...]
actions = np.array(['hello', 'thanks', 'drink']) # Unique labels
label_map = {label:num for num, label in enumerate(actions)}

# Convert labels to numbers (0, 1, 2) and then to "One-Hot" encoding
# 'hello' becomes [1, 0, 0], 'thanks' becomes [0, 1, 0], etc.
y = to_categorical([label_map[label] for label in labels]).astype(int)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)



model = Sequential()
# Input shape: (Number of frames, Number of landmarks per frame)
# Example: (30, 126) for 30 frames and 21 landmarks * 3 coords * 2 hands
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# The last layer must have the same number of units as your total words
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])




model.fit(X_train, y_train, epochs=200, callbacks=[tensorboard_callback])