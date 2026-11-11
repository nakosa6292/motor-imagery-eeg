from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.signal import butter, sosfilt
import numpy as np

# Load dataset
dataset = BNCI2014001()
paradigm = MotorImagery(n_classes=4)
X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Bandpass filter EEG in mu + beta band (8–30 Hz)
def bandpass(data, low=8, high=30, fs=250):
    sos = butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
    return sosfilt(sos, data, axis=-1)

X_filtered = bandpass(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

# Apply CSP
csp = CSP(n_components=12, reg=None, log=True, transform_into='average_power')
X_train_csp = csp.fit_transform(X_train, y_train)
X_test_csp = csp.transform(X_test)

# Feature selection using mutual information
selector = SelectKBest(mutual_info_classif, k=6)
X_train_sel = selector.fit_transform(X_train_csp, y_train)
X_test_sel = selector.transform(X_test_csp)

# Train and test an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train_sel, y_train)
y_pred = clf.predict(X_test_sel)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy with CSP + Feature Selection + SVM: {acc:.2f}")