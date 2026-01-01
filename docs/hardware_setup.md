# 🔧 Hardware Setup Guide

This guide explains how to assemble and connect the physical hardware for the **Smart Recycling Detection System**.

## 📦 Components Required

| Component | Quantity | Description |
|-----------|----------|-------------|
| **ESP32 DevKit V1** | 1 | Main microcontroller for servo control. |
| **Servo Motors** | 3 | MG996R (Recommended) or SG90 for the gates. |
| **Power Supply** | 1 | 5V 3A+ (External power for servos is highly recommended). |
| **Jumper Wires** | Varies | For connections between ESP32 and Servos. |
| **Recycling Bins** | 3 | Physical bins or a sorting mechanism. |

---

## 🔌 Pin Connections (ESP32)

Connect the signal wires of your servos to the following GPIO pins on the ESP32:

| Material Type | Servo ID | ESP32 GPIO Pin | Bin Description |
|---------------|----------|----------------|-----------------|
| **Bottle Glass** | 1 | **GPIO 18** | Glass Sorting Bin |
| **Bottle Plastic** | 2 | **GPIO 19** | Plastic Sorting Bin |
| **Tin Can** | 3 | **GPIO 21** | Metal/Can Sorting Bin |

### ⚠️ Important Wiring Notes:
1. **Common Ground**: Ensure the ESP32 Ground (GND) is connected to the Power Supply Ground.
2. **Servo Power**: **Do NOT** power 3 servos directly from the ESP32 5V pin. It may cause the ESP32 to brown out or overheat. Use an external 5V power source.
3. **Signal Voltage**: ESP32 outputs 3.3V logic, which is usually sufficient for most 5V servos to trigger.

---

## 🛠️ Software Setup (MCU)

1. **Firmware**: The ESP32 code is located at [`docs/mcu/esp32_servo.ino`](mcu/esp32_servo.ino).
2. **Library**: You will need the **ESP32Servo** library installed in your Arduino IDE.
3. **Upload**:
   - Open `esp32_servo.ino` in Arduino IDE.
   - Select your ESP32 board and COM port.
   - Click **Upload**.

---

## 📟 Communication Protocol

The Python application communicates with the ESP32 via Serial at **115200 Baud**.

**Command Format:** `S<ID>:<ANGLE>\n`
- `S`: Start byte
- `<ID>`: Servo ID (1, 2, or 3)
- `:`: Separator
- `<ANGLE>`: Target angle (0-180)
- `\n`: Newline terminator

**Example:** `S1:90\n` would move the Glass Bin servo to 90 degrees.

---

## 🧪 Testing Hardware

Once connected, you can test the hardware using the system's "Real Hardware" mode:

1. Connect the ESP32 to your computer via USB.
2. Identify the COM port (e.g., `COM3` on Windows).
3. Update `config/settings.py` or your `.env` file with the correct port.
4. Run the system in Real mode:
   - Windows: `manage.bat` -> Option 3
   - Linux/macOS: `make real`

> [!TIP]
> Use the **Simulator** mode first to ensure the software logic is working before testing with physical servos.
