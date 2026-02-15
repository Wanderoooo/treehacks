"""
Jetson: lightweight bike detection + video streaming to laptop.

Streams camera frames over TCP. NanoOWL runs in a background thread as a bike
trigger. Laptop receives frames + detection flag and handles recording/processing.

Runs inside the jetson-containers NanoOWL container:

    jetson-containers run --workdir /opt/nanoowl \
      -v ~/detect_and_record.py:/opt/nanoowl/detect_and_record.py \
      --device /dev/video0 \
      $(autotag nanoowl) \
      python3 detect_and_record.py
"""

import cv2
import time
import struct
import socket
import threading
import numpy as np
from PIL import Image
from nanoowl.owl_predictor import OwlPredictor

# ── Configuration ────────────────────────────────────────────────────────────

# Stream server
STREAM_PORT = 9000
SEND_TIMEOUT = 3.0  # seconds

# NanoOWL detection
ENGINE_PATH = "/opt/nanoowl/data/owl_image_encoder_patch32.engine"
TEXT_PROMPTS = ["a bicycle", "a bike", "a person riding a bicycle"]
CONFIDENCE_THRESHOLD = 0.15

# Only run NanoOWL every Nth frame (lower = more responsive, higher = less load)
DETECT_EVERY_N = 10

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# JPEG quality (lower = less bandwidth, lower quality)
JPEG_QUALITY = 70


# ── Camera reader thread ────────────────────────────────────────────────────

class CameraReader:
    """Continuously reads from camera in a dedicated thread.
    Always holds only the latest frame, preventing OpenCV buffer buildup."""

    def __init__(self, camera_index, width, height):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open USB camera")

        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        fail_count = 0
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                fail_count = 0
            else:
                fail_count += 1
                if fail_count > 100:
                    print("[camera] Too many read failures, reopening...")
                    self.cap.release()
                    time.sleep(1.0)
                    self.cap = cv2.VideoCapture(CAMERA_INDEX)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    fail_count = 0
                else:
                    time.sleep(0.001)

    def get_frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._running = False
        self._thread.join(timeout=3)
        self.cap.release()


# ── Detection thread ─────────────────────────────────────────────────────────

class DetectionThread:
    """Runs NanoOWL inference in a background thread.
    Main loop submits frames and reads bike_detected without blocking."""

    def __init__(self, predictor, text_encodings):
        self._predictor = predictor
        self._text_encodings = text_encodings
        self._lock = threading.Lock()
        self._frame_to_detect = None
        self._event = threading.Event()
        self.bike_detected = False
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            self._event.wait(timeout=1.0)
            self._event.clear()
            if not self._running:
                break
            with self._lock:
                frame = self._frame_to_detect
                self._frame_to_detect = None
            if frame is None:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            output = self._predictor.predict(
                image=image,
                text=TEXT_PROMPTS,
                text_encodings=self._text_encodings,
                threshold=CONFIDENCE_THRESHOLD,
            )
            self.bike_detected = len(output.boxes) > 0

    def submit(self, frame):
        """Submit a frame for detection (non-blocking, drops if busy)."""
        with self._lock:
            self._frame_to_detect = frame.copy()
        self._event.set()

    def stop(self):
        self._running = False
        self._event.set()
        self._thread.join(timeout=5)


# ── Socket helpers ───────────────────────────────────────────────────────────

def configure_socket(sock):
    """Enable TCP keepalive and set send timeout (Linux/Jetson)."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 2)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(SEND_TIMEOUT)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Initialize NanoOWL
    print(f"[init] Loading NanoOWL engine from {ENGINE_PATH}")
    predictor = OwlPredictor(
        "google/owlvit-base-patch32",
        image_encoder_engine=ENGINE_PATH,
    )
    print("[init] NanoOWL loaded")

    print(f"[init] Encoding text prompts: {TEXT_PROMPTS}")
    text_encodings = predictor.encode_text(TEXT_PROMPTS)
    print("[init] Text encodings ready")

    # Camera reader thread
    print(f"[init] Opening camera index {CAMERA_INDEX}")
    camera = CameraReader(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    print(f"[init] Camera opened: {FRAME_WIDTH}x{FRAME_HEIGHT}")

    # Detection thread
    detector = DetectionThread(predictor, text_encodings)

    # Start TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", STREAM_PORT))
    server.listen(1)
    print(f"[stream] Waiting for laptop connection on port {STREAM_PORT}...")

    conn, addr = server.accept()
    configure_socket(conn)
    print(f"[stream] Laptop connected from {addr}")

    frame_idx = 0

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Submit every Nth frame to detection thread (non-blocking)
            if frame_idx % DETECT_EVERY_N == 0:
                detector.submit(frame)

            # Encode frame as JPEG
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            data = jpeg.tobytes()

            # Send: [4 bytes frame size] [1 byte bike flag] [JPEG data]
            header = struct.pack(">IB", len(data), int(detector.bike_detected))

            if conn is not None:
                try:
                    conn.sendall(header + data)
                except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
                    print("[stream] Send failed, awaiting reconnect...")
                    conn.close()
                    conn = None

            # Non-blocking reconnect poll
            if conn is None:
                server.settimeout(0.1)
                try:
                    conn, addr = server.accept()
                    configure_socket(conn)
                    print(f"[stream] Laptop reconnected from {addr}")
                except socket.timeout:
                    pass

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
    finally:
        detector.stop()
        camera.stop()
        if conn:
            conn.close()
        server.close()
        print("[main] Done")


if __name__ == "__main__":
    main()
