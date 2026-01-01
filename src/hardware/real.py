"""
Real Hardware Implementation.

This module controls the actual hardware (ESP32/Servos) via Serial or GPIO.
Currently a placeholder for actual implementation.
"""

import time
import logging
from typing import Dict, Any, Optional

try:
    import serial
except ImportError:
    serial = None

from .base import HardwareInterface
from config.constants import CLASS_TO_SERVO_MAP
from config.settings import get_config

logger = logging.getLogger("hardware.real")

class RealHardware(HardwareInterface):
    """
    Actual hardware interface implementation using Serial communication.
    """
    
    def __init__(self):
        self.config = get_config().hardware
        self.ser: Optional[serial.Serial] = None
        self.is_connected = False
        self.port = self.config.port
        self.baud_rate = self.config.baud_rate
        self.timeout = self.config.timeout
        
    def connect(self) -> bool:
        if serial is None:
            logger.error("[REAL] pyserial not installed. Please run 'pip install pyserial'")
            return False

        if self.is_connected:
            return True

        logger.info(f"[REAL] Connecting to hardware on {self.port} ({self.baud_rate} baud)...")
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            # Give ESP32 time to reset/initialize
            time.sleep(2)
            self.is_connected = True
            logger.info(f"[REAL] Connected to hardware on {self.port}")
            return True
        except Exception as e:
            logger.error(f"[REAL] Connection failed on {self.port}: {e}")
            self.is_connected = False
            return False
        
    def disconnect(self):
        if self.ser and self.ser.is_open:
            logger.info("[REAL] Disconnecting hardware...")
            self.ser.close()
        self.is_connected = False
            
    def trigger_action(self, class_name: str) -> bool:
        if not self.is_connected:
            # Try to auto-connect if enabled
            if self.config.auto_connect:
                if not self.connect():
                    return False
            else:
                logger.warning("[REAL] Cannot trigger action: Not connected")
                return False
            
        servo_config = CLASS_TO_SERVO_MAP.get(class_name)
        if not servo_config:
            logger.warning(f"[REAL] No servo mapping for class: {class_name}")
            return False
            
        servo_id = servo_config['id']
        angle = servo_config['angle_open']
        
        try:
            # Simple protocol: S<ID>:<ANGLE>\n
            command = f"S{servo_id}:{angle}\n"
            self.ser.write(command.encode('utf-8'))
            logger.info(f"[REAL] 🚀 Command sent: {command.strip()} for {class_name}")
            return True
        except Exception as e:
            logger.error(f"[REAL] Failed to send command: {e}")
            # Mark as disconnected if write fails
            self.is_connected = False
            return False
        
    def get_status(self) -> Dict[str, Any]:
        return {
            "type": "Real",
            "connected": self.is_connected,
            "port": self.port,
            "baud_rate": self.baud_rate
        }
