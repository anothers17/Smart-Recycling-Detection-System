"""
Mock Hardware Implementation.

This module provides a simulated hardware interface for development and testing
without physical hardware.
"""

import time
import logging
from typing import Dict, Any
from .base import HardwareInterface

logger = logging.getLogger("hardware.mock")

# ตรงนี้ใช้จำลองตอนที่ไม่มีบอร์ดจริงเชื่อมต่อ (ตอนเทสที่หอหรือที่ห้องเรียน)
class MockHardware(HardwareInterface):
    """
    Simulated hardware interface.
    Logs actions to console and provides visual feedback placeholders.
    """
    
    def __init__(self):
        self.is_connected = False
        self.last_action_time = 0
        self.action_count = 0
        self.class_action_counts = {}
        
    def connect(self) -> bool:
        logger.info("[MOCK] Connecting to virtual hardware...")
        time.sleep(0.5)  # Simulate connection delay
        self.is_connected = True
        logger.info("[MOCK] Virtual hardware connected.")
        return True
        
    def disconnect(self):
        if self.is_connected:
            logger.info("[MOCK] Disconnecting virtual hardware...")
            self.is_connected = False
            
    def trigger_action(self, class_name: str) -> bool:
        if not self.is_connected:
            logger.warning("[MOCK] Cannot trigger action: Hardware not connected")
            return False
            
        # Log in a format similar to a real device (inspired by user example)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [Device] Sorting {class_name.upper()}... Status: OPEN")
        logger.info(f"[MOCK] 🟢 ACTION TRIGGERED for: {class_name}")
        
        self.action_count += 1
        self.class_action_counts[class_name] = self.class_action_counts.get(class_name, 0) + 1
        self.last_action_time = time.time()
        
        # Simulate small delay for the "OPEN" state before "CLOSE" (conceptually)
        # Note: In the UI we will handle the visual duration
        return True
        
    def get_status(self) -> Dict[str, Any]:
        return {
            "type": "Mock",
            "connected": self.is_connected,
            "total_actions": self.action_count,
            "class_counts": self.class_action_counts,
            "last_action_timestamp": self.last_action_time
        }
