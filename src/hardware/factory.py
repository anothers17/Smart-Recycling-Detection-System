"""
Hardware Factory.

Creates the appropriate hardware interface instance based on configuration.
"""

import logging
from .base import HardwareInterface
from .mock import MockHardware
from .real import RealHardware
from config.settings import get_config

logger = logging.getLogger("hardware.factory")

def get_hardware_interface() -> HardwareInterface:
    """
    Factory function to return the configured hardware interface.
    
    Returns:
        HardwareInterface: Instance of RealHardware or MockHardware.
    """
    config = get_config()
    
    if config.has_hardware:
        logger.info("Initializing REAL hardware interface...")
        return RealHardware()
    else:
        logger.info("Initializing MOCK hardware interface...")
        return MockHardware()
