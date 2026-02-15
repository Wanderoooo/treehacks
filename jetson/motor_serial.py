"""
Motor controller over serial USB to Arduino.
Runs on Jetson inside the NanoOWL container.

Usage:
    from motor_serial import MotorDriver

    motor = MotorDriver()  # auto-finds Arduino
    motor.arm()
    motor.forward(50)
    motor.turn_left(50, ratio=0.3)
    motor.stop()
    motor.close()
"""

import time
import serial
import glob


def find_arduino(preferred="/dev/ttyACM0"):
    """Find Arduino serial port."""
    # Try preferred first
    try:
        s = serial.Serial(preferred, 115200, timeout=1)
        s.close()
        return preferred
    except (serial.SerialException, OSError):
        pass
    # Scan common paths
    for pattern in ["/dev/ttyACM*", "/dev/ttyUSB*"]:
        ports = glob.glob(pattern)
        if ports:
            return ports[0]
    return None


class MotorDriver:
    """Sends motor commands to Arduino over serial."""

    def __init__(self, port=None, baud=115200):
        if port is None:
            port = find_arduino()
            if port is None:
                raise RuntimeError("Arduino not found. Check USB connection.")

        self.ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)  # Arduino resets on serial connect
        # Drain any startup messages
        while self.ser.in_waiting:
            print(f"[motor] {self.ser.readline().decode().strip()}")
        print(f"[motor] Connected to Arduino on {port}")

    def _send(self, cmd):
        """Send command and read response."""
        self.ser.write(f"{cmd}\n".encode())
        self.ser.flush()
        resp = self.ser.readline().decode().strip()
        if resp:
            print(f"[motor] {resp}")
        return resp

    def arm(self):
        """Arm both ESCs. Blocks for ~3 seconds."""
        print("[motor] Arming ESCs...")
        self._send("A")
        # Wait for ARMED response (arm takes 3s on Arduino side)
        resp = self.ser.readline().decode().strip()
        if resp:
            print(f"[motor] {resp}")

    def set_motors(self, left, right):
        """Set left and right motor speeds (0-100)."""
        left = max(0, min(100, int(left)))
        right = max(0, min(100, int(right)))
        self._send(f"M{left},{right}")

    def forward(self, speed=50):
        self.set_motors(speed, speed)

    def turn_left(self, speed=50, ratio=0.3):
        self.set_motors(int(speed * ratio), speed)

    def turn_right(self, speed=50, ratio=0.3):
        self.set_motors(speed, int(speed * ratio))

    def stop(self):
        self._send("S")

    def close(self):
        self.stop()
        self.ser.close()
