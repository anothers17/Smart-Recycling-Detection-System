"""
Hardware Interface Abstraction.

This module defines the abstract base class for hardware interaction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class HardwareInterface(ABC):
    """Abstract base class for hardware control."""
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the hardware.
        
        Returns:
            True if connection successful, False otherwise.
        """
        pass
        
    @abstractmethod
    def disconnect(self):
        """Close the connection to the hardware."""
        pass
        
    @abstractmethod
    def trigger_action(self, class_name: str) -> bool:
        """
        Trigger a hardware action based on the detected object class.
        
        Args:
            class_name: The name of the detected object class.
            
        Returns:
            True if action triggered successfully, False otherwise.
        """
        pass
        
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get current hardware status.
        
        Returns:
            Dictionary containing status information.
        """
        pass
