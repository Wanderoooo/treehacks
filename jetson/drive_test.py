"""
Test script: arm ESCs and drive straight.
Run inside the NanoOWL container on Jetson.

    jetson-containers run --workdir /opt/nanoowl \
      -v ~/motor_serial.py:/opt/nanoowl/motor_serial.py \
      -v ~/drive_test.py:/opt/nanoowl/drive_test.py \
      --device /dev/ttyACM0 \
      $(autotag nanoowl) \
      python3 drive_test.py

Or without container (if pyserial is installed):
    sudo python3 drive_test.py
"""

import time
from motor_serial import MotorDriver


def main():
    motor = MotorDriver()

    try:
        motor.arm()
        input("ESCs armed. Press Enter to drive forward (Ctrl+C to abort)...")

        print("Driving forward at 15% for 3 seconds...")
        motor.forward(speed=15)
        time.sleep(3)

        print("Stopping.")
        motor.stop()
        time.sleep(1)

    except KeyboardInterrupt:
        print("\nEmergency stop!")
    finally:
        motor.close()
        print("Done.")


if __name__ == "__main__":
    main()
