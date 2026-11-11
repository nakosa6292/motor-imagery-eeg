from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset
dataset = BNCI2014001()
paradigm = MotorImagery(n_classes=4)
X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

# Encode string labels to integers
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CSP + SVM pipeline
csp = CSP(n_components=6, reg=None, log=True, transform_into='average_power')

svm = SVC(kernel='rbf', C=1, gamma='scale')  # RBF kernel SVM
clf = Pipeline([('csp', csp), ('svm', svm)])

# Fit and evaluate
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Test accuracy with CSP + SVM: {acc:.2f}")