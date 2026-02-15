import Jetson.GPIO as GPIO
import time


class MotorController:
    """Controls a single brushless motor via ESC PWM signal."""

    def __init__(self, pin, freq=50):
        self.pin = pin
        self.freq = freq
        GPIO.setup(pin, GPIO.OUT)
        self.pwm = GPIO.PWM(pin, freq)
        self.pwm.start(0)

    def set_throttle(self, percent):
        """Set motor throttle 0-100%. Maps to 1000-2000us pulse width."""
        percent = max(0.0, min(100.0, float(percent)))
        duty = 5.0 + (percent / 100.0) * 5.0
        self.pwm.ChangeDutyCycle(duty)

    def arm(self):
        """Arm ESC by holding minimum throttle for 3 seconds."""
        print(f"  Arming ESC on pin {self.pin}...")
        self.set_throttle(0)
        time.sleep(3)
        print(f"  ESC on pin {self.pin} armed.")

    def stop(self):
        self.set_throttle(0)

    def cleanup(self):
        self.pwm.stop()


class DifferentialDrive:
    """Differential drive controller for 2 brushless motors.

    Wiring (Jetson Orin Nano 40-pin header):
        ESC #1 (left):  signal=Pin 32 (PWM0), GND=Pin 30
        ESC #2 (right): signal=Pin 33 (PWM1), GND=Pin 34
    """

    def __init__(self, left_pin=32, right_pin=33):
        GPIO.setmode(GPIO.BOARD)
        self.left = MotorController(left_pin)
        self.right = MotorController(right_pin)

    def arm(self):
        """Arm both ESCs. Wait for confirmation beeps."""
        print("Arming ESCs...")
        self.left.set_throttle(0)
        self.right.set_throttle(0)
        time.sleep(3)
        print("Both ESCs armed.")

    def forward(self, speed=50):
        """Drive forward. speed: 0-100%."""
        self.left.set_throttle(speed)
        self.right.set_throttle(speed)

    def turn_left(self, speed=50, ratio=0.3):
        """Turn left by slowing the left motor."""
        self.left.set_throttle(speed * ratio)
        self.right.set_throttle(speed)

    def turn_right(self, speed=50, ratio=0.3):
        """Turn right by slowing the right motor."""
        self.left.set_throttle(speed)
        self.right.set_throttle(speed * ratio)

    def stop(self):
        self.left.stop()
        self.right.stop()

    def cleanup(self):
        self.stop()
        self.left.cleanup()
        self.right.cleanup()
        GPIO.cleanup()
