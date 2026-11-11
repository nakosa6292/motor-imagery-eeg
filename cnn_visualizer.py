import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras.models import load_model

# Load trained CNN model and EEG test data
model = load_model("eegnet_model.h5")
X_test = np.load("X_test.npy")

# Predict using the model
predictions = model.predict(X_test)
labels = np.argmax(predictions, axis=1)

# Map numeric predictions to labels
label_map = {0: "left_hand", 1: "right_hand", 2: "feet", 3: "tongue"}

# Setup the plot
fig, ax = plt.subplots()
icon, = ax.plot([], [], 'o', markersize=20)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("EEG Prediction Visualization")

# Define fixed positions for each imagined movement
positions = {
    "left_hand": (2, 5),
    "right_hand": (8, 5),
    "feet": (5, 2),
    "tongue": (5, 8),
}

# Update function for animation
def update(i):
    label = label_map[labels[i]]
    x, y = positions[label]
    icon.set_data([x], [y])
    ax.set_title(f"Predicted: {label}")
    return icon,

# Run animation
ani = animation.FuncAnimation(fig, update, frames=len(labels), interval=800, blit=True)
plt.show()