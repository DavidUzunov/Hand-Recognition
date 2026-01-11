import numpy as np

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

# Usage during your data loading loop:
processed_sequences = []
for seq in sequences:
    fixed_seq = adjust_sequence_length(seq, target_len=40)
    processed_sequences.append(fixed_seq)

X = np.array(processed_sequences) # This will now work! Shape: (samples, 40, 126)