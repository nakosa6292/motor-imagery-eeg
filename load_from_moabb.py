from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
import numpy as np

# Load dataset: BCI Competition IV 2a (Graz)
dataset = BNCI2014001()

# Define the motor imagery task
paradigm = MotorImagery(n_classes=4)  # left, right, foot, tongue

# Load data from subject 1
X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

# Print shapes
print("EEG shape:", X.shape)        # trials × channels × time
print("Labels:", np.unique(labels)) # should be 0–3 for 4 classes



import numpy as np
from sklearn.preprocessing import LabelEncoder

# Convert string labels to integers
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

print("Encoded labels:", np.unique(y))


#filtering for EEG data mu and beta bands
from scipy.signal import butter, sosfilt

# Bandpass filter: 8–30 Hz
def bandpass(data, low=8, high=30, fs=250):
    sos = butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
    return sosfilt(sos, data, axis=-1)

# Apply filter to all trials
X_filtered = bandpass(X)
print("Filtered EEG shape:", X_filtered.shape)


# Extract log-variance features per trial
def extract_logvar(trials):
    return np.log(np.var(trials, axis=2))  # shape: (n_trials, n_channels)

X_features = extract_logvar(X_filtered)
print("Feature shape:", X_features.shape)  # Should be (576, 22)



#train a classifier using use Linear Discriminant Analysis (LDA)
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Train classifier
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.2f}")