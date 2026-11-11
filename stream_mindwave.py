import serial
import time

# Try your actual port here
PORT = "/dev/tty.MindWaveMobile"
BAUDRATE = 57600

print(f"🔌 Connecting to {PORT} at {BAUDRATE} baud...")
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
except Exception as e:
    print("❌ Failed to open serial port:", e)
    exit(1)

# Send init command: 0xC0 0x00 (this tells the headset to start streaming)
ser.write(bytes([0xC0, 0x00]))
print("📡 Sent init command. Listening for data...")

def read_packet():
    # Wait for sync bytes
    while True:
        if ser.read() == b'\xaa':
            if ser.read() == b'\xaa':
                payload_len = ser.read()[0]
                payload = ser.read(payload_len)
                checksum = ser.read()[0]
                print("📥 Packet received:", payload.hex())
                break

while True:
    try:
        read_packet()
    except KeyboardInterrupt:
        print("🛑 Exiting...")
        break