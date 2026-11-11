import serial
import time
import struct
import matplotlib.pyplot as plt
from collections import deque

# Serial settings
PORT = "/dev/cu.MindWaveMobile"
BAUDRATE = 57600

# EEG data queues for plotting
history_len = 100
attention_history = deque([0]*history_len, maxlen=history_len)
meditation_history = deque([0]*history_len, maxlen=history_len)
signal_history = deque([0]*history_len, maxlen=history_len)

# Open serial port
print(f"🔌 Connecting to {PORT} at {BAUDRATE} baud...")
ser = serial.Serial(PORT, BAUDRATE, timeout=1)

# Init streaming
ser.write(bytes([0xC0, 0x00]))
print("📡 Streaming EEG data...")

def read_packet():
    while True:
        if ser.read() == b'\xaa':
            if ser.read() == b'\xaa':
                payload_len = ser.read()[0]
                payload = ser.read(payload_len)
                checksum = ser.read()[0]
                return payload

# Matplotlib setup
plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot(attention_history, label="Attention")
line2, = ax.plot(meditation_history, label="Meditation")
line3, = ax.plot(signal_history, label="Signal Quality")
ax.set_ylim(0, 110)
ax.set_title("MindWave EEG - Real-Time")
ax.legend()

while True:
    try:
        payload = read_packet()
        i = 0
        while i < len(payload):
            code = payload[i]
            i += 1

            if i >= len(payload):
                break

            if code == 0x02 and i < len(payload):  # Signal
                val = payload[i]
                signal_history.append(100 - val)  # invert so higher = better
                i += 1

            elif code == 0x04 and i < len(payload):  # Attention
                val = payload[i]
                attention_history.append(val)
                i += 1

            elif code == 0x05 and i < len(payload):  # Meditation
                val = payload[i]
                meditation_history.append(val)
                i += 1

            else:
                if code >= 0x80 and i < len(payload):
                    i += 1 + payload[i]  # skip extended code
                else:
                    i += 1

        # Update plot
        line1.set_ydata(attention_history)
        line2.set_ydata(meditation_history)
        line3.set_ydata(signal_history)
        line1.set_xdata(range(len(attention_history)))
        line2.set_xdata(range(len(meditation_history)))
        line3.set_xdata(range(len(signal_history)))
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Exiting visualizer...")
        break