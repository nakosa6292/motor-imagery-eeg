import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras.models import load_model

# Load model and EEG test data
model = load_model("eegnet_model.h5")
X_test = np.load("X_test.npy")
predictions = model.predict(X_test)
labels = np.argmax(predictions, axis=1)

# EEG label → arm pose (target joint angles in radians)
label_to_pose = {
    0: (np.pi / 3, np.pi / 3),     # left_hand
    1: (np.pi / 3, -np.pi / 3),    # right_hand
    2: (np.pi / 2, 0),             # feet (point down)
    3: (-np.pi / 2, np.pi / 2),    # tongue (curl up)
}

# Arm link lengths
L1, L2 = 3, 2

# Start with neutral position
current_theta1, current_theta2 = 0.0, 0.0

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=5, marker='o')
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.set_title("🦾 Prosthetic Arm Simulation")

label_text = ax.text(-5.5, 5.5, '', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
desc_text = ax.text(-5.5, 4.5, '', fontsize=10, color='gray')

# Tweening helper
def smooth_step(current, target, alpha=0.2):
    return current + alpha * (target - current)

def update(i):
    global current_theta1, current_theta2
    label = labels[i % len(labels)]
    target_theta1, target_theta2 = label_to_pose[label]

    # Smooth transition
    current_theta1 = smooth_step(current_theta1, target_theta1)
    current_theta2 = smooth_step(current_theta2, target_theta2)

    # Joint positions
    x0, y0 = 0, 0
    x1 = L1 * np.cos(current_theta1)
    y1 = L1 * np.sin(current_theta1)
    x2 = x1 + L2 * np.cos(current_theta1 + current_theta2)
    y2 = y1 + L2 * np.sin(current_theta1 + current_theta2)

    # Draw
    line.set_data([x0, x1, x2], [y0, y1, y2])

    # Label mappings
    label_names = ["Left Hand", "Right Hand", "Feet", "Tongue"]
    descriptions = [
        "Reach upper-left",
        "Reach upper-right",
        "Point straight down",
        "Curl upward"
    ]
    label_text.set_text(f"🧠 Predicted: {label_names[label]}")
    desc_text.set_text(f"🦾 Prosthetic Command: {descriptions[label]}")

    return line, label_text, desc_text

ani = animation.FuncAnimation(fig, update, frames=200, interval=300, blit=True)
plt.show()