/*
 * Smart Recycling Detection System - ESP32 Servo Control
 * 
 * This sketch receives serial commands from the Python application
 * and controls three servo motors to sort recycling materials.
 * 
 * Command format: S<ID>:<ANGLE>\n
 * Example: S1:90\n (Move servo 1 to 90 degrees)
 * 
 * Setup:
 * - Servo 1 (Bottle Glass): Pin 18 (Glass Bin)
 * - Servo 2 (Bottle Plastic): Pin 19 (Plastic Bin)
 * - Servo 3 (Tin Can): Pin 21 (Can Bin)
 * 
 * Note: ถ้าบอร์ด ESP32 ร้อนเกินไป ให้เช็คไฟเลี้ยง Servo แยกนะครับ (Electronic SUT)
 */

#include <ESP32Servo.h>

// Define Servo pins
const int SERVO_1_PIN = 18;
const int SERVO_2_PIN = 19;
const int SERVO_3_PIN = 21;

// Servo objects
Servo servo1;
Servo servo2;
Servo servo3;

// Neutral/Closed position
const int CLOSED_ANGLE = 0;
// Open position (default, can be overridden by command)
const int OPEN_ANGLE = 90;
// Time to keep the gate open (ms)
const int OPEN_DELAY = 2000;

void setup() {
  Serial.begin(115200);
  
  // Attach servos
  servo1.attach(SERVO_1_PIN);
  servo2.attach(SERVO_2_PIN);
  servo3.attach(SERVO_3_PIN);
  
  // Initial positions
  servo1.write(CLOSED_ANGLE);
  servo2.write(CLOSED_ANGLE);
  servo3.write(CLOSED_ANGLE);
  
  Serial.println("ESP32 Recycling System Ready!");
  Serial.println("Protocol: S<ID>:<ANGLE>\\n");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.startsWith("S")) {
      int colonIndex = input.indexOf(':');
      if (colonIndex != -1) {
        // Extract ID and Angle
        int id = input.substring(1, colonIndex).toInt();
        int angle = input.substring(colonIndex + 1).toInt();
        
        Serial.print("Received Command for Servo ");
        Serial.print(id);
        Serial.print(" to ");
        Serial.print(angle);
        Serial.println(" deg");
        
        // Execute movement
        triggerServo(id, angle);
      }
    }
  }
}

void triggerServo(int id, int angle) {
  Servo* targetServo = NULL;
  
  switch(id) {
    case 1: targetServo = &servo1; break;
    case 2: targetServo = &servo2; break;
    case 3: targetServo = &servo3; break;
    default:
      Serial.println("Error: Invalid Servo ID");
      return;
  }
  
  // Open
  targetServo->write(angle);
  delay(OPEN_DELAY);
  // Auto-close
  targetServo->write(CLOSED_ANGLE);
  Serial.println("Action completed: Open-Close cycle done.");
}
