import numpy as np
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, SeparableConv2D, BatchNormalization,
    Activation, AveragePooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.utils import to_categorical

# Load data
dataset = BNCI2014001()
paradigm = MotorImagery(n_classes=4)
X, labels, _ = paradigm.get_data(dataset=dataset, subjects=[1])
print("EEG shape:", X.shape)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
y_cat = to_categorical(y)

# Reshape input: (samples, channels, time, 1)
X = X.astype(np.float32)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)


# EEGNet-like architecture
model = Sequential([
    Conv2D(16, (1, 64), padding='same', input_shape=X_train.shape[1:], use_bias=False),
    BatchNormalization(),
    DepthwiseConv2D((X.shape[1], 1), use_bias=False, depth_multiplier=2, padding='valid'),
    BatchNormalization(),
    Activation('elu'),
    AveragePooling2D((1, 4)),
    Dropout(0.25),
    SeparableConv2D(32, (1, 16), use_bias=False, padding='same'),
    BatchNormalization(),
    Activation('elu'),
    AveragePooling2D((1, 8)),
    Dropout(0.25),
    Flatten(),
    Dense(4, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🧠 Test accuracy with CNN (EEGNet-style): {acc:.2f}")



# Save test data and model for visualization
np.save("X_test.npy", X_test)
model.save("eegnet_model.h5")