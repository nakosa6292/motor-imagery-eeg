import serial
import time

# ✅ Correct device port
PORT = "/dev/cu.MindWaveMobile"
BAUDRATE = 57600

print(f"🔌 Connecting to {PORT} at {BAUDRATE} baud...")
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
except Exception as e:
    print("❌ Failed to open serial port:", e)
    exit(1)

# 🔄 Send init command to start EEG data stream
ser.write(bytes([0xC0, 0x00]))
print("📡 Sent init command. Listening for data...")

def read_packet():
    while True:
        if ser.read() == b'\xaa':
            if ser.read() == b'\xaa':
                payload_len = ser.read()[0]
                payload = ser.read(payload_len)
                checksum = ser.read()[0]

                if checksum != (255 - (sum(payload) & 0xFF)):
                    print("❌ Invalid checksum")
                    return

                i = 0
                while i < len(payload):
                    code = payload[i]
                    i += 1

                    if code == 0x02:
                        signal = payload[i]
                        i += 1
                        print(f"📶 Signal quality (0=good): {signal}")
                    elif code == 0x04:
                        attention = payload[i]
                        i += 1
                        print(f"🧠 Attention: {attention}")
                    elif code == 0x05:
                        meditation = payload[i]
                        i += 1
                        print(f"🧘 Meditation: {meditation}")
                    elif code == 0x80:
                        raw_length = payload[i]
                        i += 1
                        raw_wave = int.from_bytes(payload[i:i+2], byteorder='big', signed=True)
                        i += 2
                        print(f"🌊 Raw EEG: {raw_wave}")
                    elif code == 0x83:
                        eeg_len = payload[i]
                        i += 1
                        eeg_data = payload[i:i+eeg_len]
                        i += eeg_len
                        print("📊 EEG Bands (μV²):")
                        bands = ['delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'midGamma']
                        for j in range(8):
                            val = int.from_bytes(eeg_data[j*3:j*3+3], 'big')
                            print(f"  {bands[j]}: {val}")
                    else:
                        print(f"❓ Unknown code {code}")
# 🧠 Stream loop
while True:
    try:
        read_packet()
    except KeyboardInterrupt:
        print("🛑 Exiting...")
        break