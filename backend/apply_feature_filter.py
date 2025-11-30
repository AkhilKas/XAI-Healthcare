import numpy as np
from sklearn.preprocessing import StandardScaler

def apply_feature_filter(g1, g0):
    all_tensors = []
    all_labels = []

    # patient data (label = 1)
    for k, tensor_data in g1.items():
        all_tensors.append(tensor_data)
        all_labels.append(1)

    # control data (label = 0)
    for k, tensor_data in g0.items():
        all_tensors.append(tensor_data)
        all_labels.append(0)

    print(f"\nTotal number of samples: {len(all_tensors)}")
    print(f"Sample shapes: {[t.shape for t in all_tensors]}")

    y = np.array(all_labels, dtype=np.int32)
    signals_only = [t[:,1:] for t in all_tensors]

    scaler = StandardScaler()

    X_concat = np.vstack(signals_only)
    X_scaled = scaler.fit_transform(X_concat)
    
    lengths = [len(seq) for seq in signals_only]

    scaled_sequences = []
    scaled_y = []
    start = 0
    for i, l in enumerate(lengths):
        if l <= 5000:
            scaled_sequences.append(X_scaled[start:start+l])
            scaled_y.append(y[i])
        start += l
    X_scaled = scaled_sequences
    y = np.array(scaled_y, dtype=np.int32)

    print(f"\nFinal X_scaled shape: {X_scaled.shape if hasattr(X_scaled, 'shape') else len(X_scaled)}")
    print(f"Final y shape: {y.shape}")
    return X_scaled, y