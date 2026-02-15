"""
Keyboard remote control for the robot. Runs on your Mac.
Sends UDP commands to the Jetson container.

Controls:
    W / Up    = forward
    S / Down  = stop
    A / Left  = turn left
    D / Right = turn right
    Q         = quit
    +/-       = increase/decrease speed

Usage:
    python keyboard_drive.py 100.73.97.1
"""

import sys
import socket
import tty
import termios

JETSON_PORT = 9001
DEFAULT_SPEED = 20
SPEED_STEP = 5
MAX_SPEED = 80

def get_key():
    """Read a single keypress (raw mode)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':  # arrow key escape sequence
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            if ch2 == '[':
                return {'A': 'w', 'B': 's', 'C': 'd', 'D': 'a'}.get(ch3, '')
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    ip = sys.argv[1] if len(sys.argv) > 1 else "100.73.97.1"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    speed = DEFAULT_SPEED

    def send(cmd):
        sock.sendto(cmd.encode(), (ip, JETSON_PORT))

    print(f"Keyboard drive → {ip}:{JETSON_PORT}")
    print(f"Speed: {speed}%")
    print()
    print("  W/Up    = forward")
    print("  S/Down  = stop")
    print("  A/Left  = turn left")
    print("  D/Right = turn right")
    print("  +/-     = speed up/down")
    print("  SPACE   = arm ESCs")
    print("  Q       = quit")
    print()

    try:
        while True:
            key = get_key().lower()

            if key == 'q':
                send("S")
                print("\nQuit.")
                break
            elif key == ' ':
                send("A")
                print("Arming ESCs...")
            elif key == 'w':
                send(f"M{speed},{speed}")
                print(f"\rForward {speed}%   ", end="", flush=True)
            elif key == 's':
                send("S")
                print(f"\rStop            ", end="", flush=True)
            elif key == 'a':
                left = int(speed * 0.3)
                send(f"M{left},{speed}")
                print(f"\rLeft {left}/{speed}%  ", end="", flush=True)
            elif key == 'd':
                right = int(speed * 0.3)
                send(f"M{speed},{right}")
                print(f"\rRight {speed}/{right}%  ", end="", flush=True)
            elif key in ('+', '='):
                speed = min(speed + SPEED_STEP, MAX_SPEED)
                print(f"\rSpeed: {speed}%     ", end="", flush=True)
            elif key in ('-', '_'):
                speed = max(speed - SPEED_STEP, 0)
                print(f"\rSpeed: {speed}%     ", end="", flush=True)

    except KeyboardInterrupt:
        send("S")
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
