# Bike Safety Dashboard

Real-time bike detection and safety monitoring system built at TreeHacks 2026. Uses a Jetson Orin Nano with NanoOWL for on-device detection, streams live video to a Mac-based dashboard, and drives an autonomous differential-drive vehicle via Arduino motor control.

## Architecture

```
┌─────────────────────────┐       TCP :9000        ┌──────────────────────────┐
│     Jetson Orin Nano     │ ───────────────────────▶│     Mac (Laptop)         │
│                          │   frames + boxes +      │                          │
│  NanoOWL detection       │   bike flag             │  stream_receiver.py      │
│  Camera capture          │                         │  ├─ light detection      │
│  Bounding box overlay    │◀─────────────────────── │  ├─ video recording      │
│  Motor command listener  │   UDP :9001             │  ├─ depth estimation     │
│                          │   keyboard drive cmds   │  └─ report generation    │
└────────┬────────────────┘                         │                          │
         │ Serial USB                                │  web/app.py (Flask)      │
         │                                           │  ├─ live MJPEG feed      │
┌────────▼────────────────┐                         │  ├─ incident dashboard   │
│       Arduino            │                         │  └─ AI chat agent        │
│  ESC PWM motor control   │                         └──────────────────────────┘
│  2x brushless motors     │
└─────────────────────────┘
```

## Components

### Jetson (`jetson/`)

| File | Purpose |
|------|---------|
| `detect_and_record.py` | Main script: camera capture, NanoOWL inference, TCP streaming, UDP motor command listener |
| `motor_serial.py` | Serial driver for Arduino motor controller |
| `motor_controller.py` | Alternative GPIO-based PWM motor control (unused with Arduino setup) |
| `drive_test.py` | Standalone motor test script |

Runs inside the NanoOWL Docker container on Jetson:

```bash
jetson-containers run --workdir /opt/nanoowl \
  -v ~/detect_and_record.py:/opt/nanoowl/detect_and_record.py \
  -v ~/motor_serial.py:/opt/nanoowl/motor_serial.py \
  --device /dev/video0 \
  --device /dev/ttyACM0 \
  $(autotag nanoowl) \
  bash -c "pip install pyserial && python3 detect_and_record.py"
```

### Arduino (`arduino/motor_controller/`)

Controls 2 brushless ESCs via PWM. Serial protocol (115200 baud):

| Command | Action |
|---------|--------|
| `A\n` | Arm both ESCs (3s blocking) |
| `M<left>,<right>\n` | Set motor speeds 0-100 (e.g. `M60,60`) |
| `S\n` | Stop both motors |

### Mac / Laptop

| File | Purpose |
|------|---------|
| `stream_receiver.py` | Receives TCP stream from Jetson, runs light detection on bike crops, records 5s clips, generates reports, runs depth estimation |
| `web/app.py` | Flask dashboard with live feed, incident table, AI chat |
| `keyboard_drive.py` | Terminal keyboard controller — sends UDP drive commands to Jetson |

### Streaming Protocol (TCP :9000)

Each frame sent as:
```
[4B frame_size][4B boxes_size][1B bike_flag][JPEG data][boxes JSON]
```

### Dashboard (`web/`)

Flask app serving at `http://localhost:8081`:
- Live MJPEG camera feed from Jetson (`/api/live`)
- Incident table with annotated video, depth video, and JSON reports
- AI chat agent for querying bike safety data

## Setup & Usage

### 1. Jetson Orin Nano

SCP files to the Jetson:
```bash
scp jetson/detect_and_record.py jetson/motor_serial.py jetson@100.73.97.1:~/
```

SSH in and run the Docker container (see command above).

### 2. Arduino

Flash `arduino/motor_controller/motor_controller.ino` via Arduino IDE. Connect to Jetson via USB serial.

### 3. Mac Dashboard

```bash
pip install -r requirements.txt
python web/app.py <JETSON_TAILSCALE_IP>
# e.g. python web/app.py 100.73.97.1
```

### 4. Keyboard Driving (optional)

```bash
python keyboard_drive.py <JETSON_TAILSCALE_IP>
```

Controls: `W`/`S`/`A`/`D` or arrow keys, `+`/`-` speed, `SPACE` arm ESCs, `Q` quit.

## Wiring

### ESC → Jetson 40-pin Header (if using GPIO, pins 32/33)
### ESC → Arduino (if using serial, pins 9/10)

| ESC Wire | Arduino Pin |
|----------|-------------|
| Left signal | Pin 9 |
| Right signal | Pin 10 |
| GND | Arduino GND |
| VCC (red) | **DISCONNECTED** |

Battery connects directly to ESC power inputs (not through Arduino or Jetson).

## Project Structure

```
treehacks/
├── arduino/motor_controller/   # Arduino ESC controller
├── jetson/                     # Jetson Orin Nano scripts
│   ├── detect_and_record.py    # Camera + NanoOWL + streaming
│   ├── motor_serial.py         # Arduino serial driver
│   ├── motor_controller.py     # GPIO PWM alternative
│   └── drive_test.py           # Motor test
├── src/                        # CV pipeline modules
│   ├── detector.py             # YOLO bike detection
│   ├── light_detector.py       # Bike light detection
│   ├── depth_estimator.py      # Depth estimation
│   ├── annotator.py            # Video annotation
│   ├── report_generator.py     # JSON report generation
│   └── ...
├── web/                        # Flask dashboard
│   ├── app.py
│   ├── templates/
│   └── static/
├── stream_receiver.py          # Mac-side stream receiver
├── keyboard_drive.py           # Keyboard motor control
├── agent.py                    # AI chat agent
├── config.yaml                 # Pipeline configuration
└── requirements.txt
```

## Hardware

- Jetson Orin Nano (JetPack OS, Tailscale VPN)
- USB camera (640x480)
- IMU (I2C on pins 1, 3, 5, 6)
- Arduino Uno/Nano (serial USB to Jetson)
- 2x THOR 2830 Brushless Motors (1120KV, 51:1 gear ratio)
- 2x ESCs (standard RC, 50Hz PWM)
- Battery (~15V)
