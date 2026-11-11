import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras.models import load_model

# Load model and EEG test data
model = load_model("eegnet_model.h5")
X_test = np.load("X_test.npy")
predictions = model.predict(X_test)
labels = np.argmax(predictions, axis=1)

label_names = ["Left Hand", "Right Hand", "Feet", "Tongue"]
label_actions = [
    "Raise left arm",
    "Raise right arm",
    "Kick both legs",
    "Stick out tongue"
]

fig, ax = plt.subplots()
ax.set_xlim(-4, 4)
ax.set_ylim(-1, 7)
ax.set_aspect('equal')
ax.axis('off')

# Stick figure
head = plt.Circle((0, 6), 0.5, fill=False, lw=2)
body, left_arm, right_arm = [ax.plot([], [], lw=2)[0] for _ in range(3)]
left_leg, right_leg = [ax.plot([], [], lw=2)[0] for _ in range(2)]
tongue = ax.plot([], [], color='red', lw=2)[0]

label_text = ax.text(-3.5, 6.5, '', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
desc_text = ax.text(-3.5, 6.0, '', fontsize=10, color='gray')

ax.add_patch(head)

def update(i):
    label = labels[i % len(labels)]

    body.set_data([0, 0], [5, 6])

    left_arm_coords = ([-1, 0], [5.5, 5])
    right_arm_coords = ([1, 0], [5.5, 5])
    left_leg_coords = ([-0.5, 0], [4, 5])
    right_leg_coords = ([0.5, 0], [4, 5])
    tongue_coords = ([0, 0], [6, 6])

    if label == 0:
        left_arm_coords = ([-1, 0], [6, 5])
    elif label == 1:
        right_arm_coords = ([1, 0], [6, 5])
    elif label == 2:
        left_leg_coords = ([-0.5, 0], [3.5, 5])
        right_leg_coords = ([0.5, 0], [3.5, 5])
    elif label == 3:
        tongue_coords = ([0, 0], [6, 5.5])

    left_arm.set_data(*left_arm_coords)
    right_arm.set_data(*right_arm_coords)
    left_leg.set_data(*left_leg_coords)
    right_leg.set_data(*right_leg_coords)
    tongue.set_data(*tongue_coords)

    label_text.set_text(f"🧠 Predicted: {label_names[label]}")
    desc_text.set_text(f"🦾 Action: {label_actions[label]}")

    return body, left_arm, right_arm, left_leg, right_leg, tongue, label_text, desc_text

ani = animation.FuncAnimation(fig, update, frames=len(labels), interval=500, blit=True)
plt.show()