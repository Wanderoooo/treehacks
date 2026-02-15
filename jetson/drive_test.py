"""
Standalone test script for brushless motor control via ESC PWM.

Run on Jetson Orin Nano:
    sudo python3 drive_test.py

Wiring:
    ESC #1 (left):  signal=Pin 32, GND=Pin 30
    ESC #2 (right): signal=Pin 33, GND=Pin 34
    VCC (red wire) on both ESCs: DISCONNECTED

Test sequence:
    1. Arm ESCs (3s at min throttle - listen for beeps)
    2. Forward at 15% for 3s
    3. Turn left at 15% for 2s
    4. Turn right at 15% for 2s
    5. Stop

Press Ctrl+C at any time for emergency stop.
"""

import time
from motor_controller import DifferentialDrive


def main():
    drive = DifferentialDrive(left_pin=32, right_pin=33)

    try:
        # Step 1: Arm
        drive.arm()
        input("ESCs armed. Press Enter to start test sequence (or Ctrl+C to abort)...")

        # Step 2: Forward
        print("Forward at 15% throttle...")
        drive.forward(speed=15)
        time.sleep(3)

        # Step 3: Turn left
        print("Turning left...")
        drive.turn_left(speed=15, ratio=0.3)
        time.sleep(2)

        # Step 4: Turn right
        print("Turning right...")
        drive.turn_right(speed=15, ratio=0.3)
        time.sleep(2)

        # Step 5: Stop
        print("Stopping.")
        drive.stop()
        time.sleep(1)

    except KeyboardInterrupt:
        print("\nEmergency stop!")
    finally:
        drive.cleanup()
        print("Motors stopped, GPIO cleaned up.")


if __name__ == "__main__":
    main()
