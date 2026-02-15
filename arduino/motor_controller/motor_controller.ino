/*
 * Arduino Motor Controller
 * Receives serial commands from Jetson, drives 2 brushless ESCs.
 *
 * Wiring:
 *   Pin 9  → ESC #1 signal (left motor)
 *   Pin 10 → ESC #2 signal (right motor)
 *   GND    → ESC GND (both)
 *   ESC power → battery (NOT Arduino)
 *
 * Serial protocol (115200 baud):
 *   A\n          → arm both ESCs
 *   M<L>,<R>\n   → set motor speeds (0-100), e.g. M60,60
 *   S\n          → stop both motors
 */

#include <Servo.h>

#define LEFT_PIN  9
#define RIGHT_PIN 10
#define MIN_US    1000  // ESC min (motor off)
#define MAX_US    2000  // ESC max (full throttle)
#define BAUD      115200

Servo leftESC;
Servo rightESC;

String inputBuffer = "";

void setup() {
  Serial.begin(BAUD);
  leftESC.attach(LEFT_PIN);
  rightESC.attach(RIGHT_PIN);

  // Send min throttle on startup (safe default)
  leftESC.writeMicroseconds(MIN_US);
  rightESC.writeMicroseconds(MIN_US);

  Serial.println("OK:READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  char type = cmd.charAt(0);

  switch (type) {
    case 'A': // Arm
      armESCs();
      break;

    case 'M': { // Motor speeds: M<left>,<right>
      int commaIdx = cmd.indexOf(',');
      if (commaIdx < 0) {
        Serial.println("ERR:FORMAT");
        return;
      }
      int leftSpeed  = cmd.substring(1, commaIdx).toInt();
      int rightSpeed = cmd.substring(commaIdx + 1).toInt();
      setMotors(leftSpeed, rightSpeed);
      break;
    }

    case 'S': // Stop
      setMotors(0, 0);
      Serial.println("OK:STOP");
      break;

    default:
      Serial.println("ERR:UNKNOWN");
      break;
  }
}

void armESCs() {
  Serial.println("OK:ARMING");
  leftESC.writeMicroseconds(MIN_US);
  rightESC.writeMicroseconds(MIN_US);
  delay(3000);
  Serial.println("OK:ARMED");
}

void setMotors(int leftPct, int rightPct) {
  leftPct  = constrain(leftPct,  0, 100);
  rightPct = constrain(rightPct, 0, 100);

  int leftUs  = map(leftPct,  0, 100, MIN_US, MAX_US);
  int rightUs = map(rightPct, 0, 100, MIN_US, MAX_US);

  leftESC.writeMicroseconds(leftUs);
  rightESC.writeMicroseconds(rightUs);

  Serial.print("OK:M");
  Serial.print(leftPct);
  Serial.print(",");
  Serial.println(rightPct);
}
