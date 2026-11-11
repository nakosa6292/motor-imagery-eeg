import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras.models import load_model

# Load trained model and test data
model = load_model("eegnet_model.h5")
X_test = np.load("X_test.npy")
predictions = model.predict(X_test)
labels = np.argmax(predictions, axis=1)

# Define the prosthetic limb: a simple 2-joint arm
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 2)
arm_line, = ax.plot([], [], lw=5)
ax.set_title("Prosthetic Response Simulation")

# Arm base (shoulder)
origin = np.array([0, 0])

# Map movement classes to angles
angle_map = {
    0: np.pi / 2,     # left hand → 90°
    1: 0,             # right hand → 0°
    2: -np.pi / 2,    # feet → -90°
    3: np.pi / 4      # tongue → 45°
}

# Update function for animation
def update(i):
    angle = angle_map[labels[i]]
    end = origin + np.array([np.cos(angle), np.sin(angle)])
    arm_line.set_data([origin[0], end[0]], [origin[1], end[1]])
    ax.set_title(f"Prosthetic response: {['left', 'right', 'feet', 'tongue'][labels[i]]}")
    return arm_line,

# Animate
ani = animation.FuncAnimation(fig, update, frames=len(labels), interval=800, blit=True)
plt.show()