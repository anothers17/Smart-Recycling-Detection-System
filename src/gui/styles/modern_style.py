"""
Modern styling for Smart Recycling Detection System GUI.

This module provides modern, professional styling for the PyQt5 interface
with themes, color schemes, and responsive design elements.
"""

from typing import Dict, Any
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import QApplication

from config.logging_config import get_logger

logger = get_logger('gui')


class ModernColors:
    """Modern color palette for the application."""
    
    # Primary colors
    PRIMARY = "#2196F3"           # Blue
    PRIMARY_DARK = "#1976D2"      # Dark Blue
    PRIMARY_LIGHT = "#BBDEFB"     # Light Blue
    
    # Secondary colors
    SECONDARY = "#4CAF50"         # Green
    SECONDARY_DARK = "#388E3C"    # Dark Green
    SECONDARY_LIGHT = "#C8E6C9"   # Light Green
    
    # Accent colors
    ACCENT = "#FF9800"            # Orange
    ACCENT_DARK = "#F57C00"       # Dark Orange
    ACCENT_LIGHT = "#FFE0B2"      # Light Orange
    
    # Status colors
    SUCCESS = "#4CAF50"           # Green
    WARNING = "#FF9800"           # Orange
    ERROR = "#F44336"             # Red
    INFO = "#2196F3"              # Blue
    
    # Neutral colors
    BACKGROUND = "#F5F5F5"        # Light Gray
    SURFACE = "#FFFFFF"           # White
    ON_SURFACE = "#212121"        # Dark Gray
    ON_BACKGROUND = "#424242"     # Medium Gray
    
    # Text colors
    TEXT_PRIMARY = "#212121"      # Dark Gray
    TEXT_SECONDARY = "#757575"    # Medium Gray
    TEXT_DISABLED = "#BDBDBD"     # Light Gray
    TEXT_ON_PRIMARY = "#FFFFFF"   # White
    
    # Border colors
    BORDER = "#E0E0E0"            # Light Gray
    BORDER_FOCUS = "#2196F3"      # Blue
    BORDER_ERROR = "#F44336"      # Red


class ModernFonts:
    """Modern font definitions."""
    
    @staticmethod
    def get_font(size: int = 10, weight: str = "normal", family: str = "Segoe UI") -> QFont:
        """
        Get a QFont with modern styling.
        
        Args:
            size: Font size
            weight: Font weight ("normal", "bold", "light")
            family: Font family
            
        Returns:
            QFont object
        """
        font = QFont(family, size)
        
        if weight == "bold":
            font.setWeight(QFont.Bold)
        elif weight == "light":
            font.setWeight(QFont.Light)
        else:
            font.setWeight(QFont.Normal)
        
        return font
    
    @staticmethod
    def heading_font(size: int = 16) -> QFont:
        """Get font for headings."""
        return ModernFonts.get_font(size, "bold")
    
    @staticmethod
    def body_font(size: int = 10) -> QFont:
        """Get font for body text."""
        return ModernFonts.get_font(size)
    
    @staticmethod
    def button_font(size: int = 10) -> QFont:
        """Get font for buttons."""
        return ModernFonts.get_font(size, "normal")
    
    @staticmethod
    def mono_font(size: int = 9) -> QFont:
        """Get monospace font for code/logs."""
        return QFont("Consolas", size)


class ModernStyleSheet:
    """Modern stylesheet generator for different themes."""
    
    @staticmethod
    def get_main_window_style() -> str:
        """Get stylesheet for main window."""
        return f"""
        QMainWindow {{
            background-color: {ModernColors.BACKGROUND};
            color: {ModernColors.TEXT_PRIMARY};
        }}
        
        QMenuBar {{
            background-color: {ModernColors.SURFACE};
            border-bottom: 1px solid {ModernColors.BORDER};
            padding: 4px;
        }}
        
        QMenuBar::item {{
            background-color: transparent;
            padding: 4px 12px;
            border-radius: 4px;
        }}
        
        QMenuBar::item:selected {{
            background-color: {ModernColors.PRIMARY_LIGHT};
        }}
        
        QStatusBar {{
            background-color: {ModernColors.SURFACE};
            border-top: 1px solid {ModernColors.BORDER};
            color: {ModernColors.TEXT_SECONDARY};
        }}
        """
    
    @staticmethod
    def get_button_style() -> str:
        """Get stylesheet for buttons."""
        return f"""
        QPushButton {{
            background-color: {ModernColors.PRIMARY};
            color: {ModernColors.TEXT_ON_PRIMARY};
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 500;
            min-height: 16px;
        }}
        
        QPushButton:hover {{
            background-color: {ModernColors.PRIMARY_DARK};
        }}
        
        QPushButton:pressed {{
            background-color: {ModernColors.PRIMARY_DARK};
            padding: 11px 19px 9px 21px;
        }}
        
        QPushButton:disabled {{
            background-color: {ModernColors.TEXT_DISABLED};
            color: {ModernColors.TEXT_SECONDARY};
        }}
        
        QPushButton.success {{
            background-color: {ModernColors.SUCCESS};
        }}
        
        QPushButton.success:hover {{
            background-color: {ModernColors.SECONDARY_DARK};
        }}
        
        QPushButton.warning {{
            background-color: {ModernColors.WARNING};
        }}
        
        QPushButton.warning:hover {{
            background-color: {ModernColors.ACCENT_DARK};
        }}
        
        QPushButton.error {{
            background-color: {ModernColors.ERROR};
        }}
        
        QPushButton.error:hover {{
            background-color: #D32F2F;
        }}
        """
    
    @staticmethod
    def get_input_style() -> str:
        """Get stylesheet for input fields."""
        return f"""
        QLineEdit {{
            background-color: {ModernColors.SURFACE};
            border: 2px solid {ModernColors.BORDER};
            border-radius: 6px;
            padding: 8px 12px;
            color: {ModernColors.TEXT_PRIMARY};
            selection-background-color: {ModernColors.PRIMARY_LIGHT};
        }}
        
        QLineEdit:focus {{
            border-color: {ModernColors.BORDER_FOCUS};
        }}
        
        QLineEdit:disabled {{
            background-color: {ModernColors.BACKGROUND};
            color: {ModernColors.TEXT_DISABLED};
        }}
        
        QTextEdit {{
            background-color: {ModernColors.SURFACE};
            border: 2px solid {ModernColors.BORDER};
            border-radius: 6px;
            padding: 8px;
            color: {ModernColors.TEXT_PRIMARY};
            selection-background-color: {ModernColors.PRIMARY_LIGHT};
        }}
        
        QTextEdit:focus {{
            border-color: {ModernColors.BORDER_FOCUS};
        }}
        """
    
    @staticmethod
    def get_label_style() -> str:
        """Get stylesheet for labels."""
        return f"""
        QLabel {{
            color: {ModernColors.TEXT_PRIMARY};
            background-color: transparent;
        }}
        
        QLabel.heading {{
            font-size: 18px;
            font-weight: bold;
            color: {ModernColors.TEXT_PRIMARY};
            margin: 10px 0px;
        }}
        
        QLabel.subheading {{
            font-size: 14px;
            font-weight: 600;
            color: {ModernColors.TEXT_SECONDARY};
            margin: 8px 0px;
        }}
        
        QLabel.caption {{
            font-size: 12px;
            color: {ModernColors.TEXT_SECONDARY};
        }}
        
        QLabel.error {{
            color: {ModernColors.ERROR};
        }}
        
        QLabel.warning {{
            color: {ModernColors.WARNING};
        }}
        
        QLabel.success {{
            color: {ModernColors.SUCCESS};
        }}
        """
    
    @staticmethod
    def get_container_style() -> str:
        """Get stylesheet for containers and frames."""
        return f"""
        QFrame {{
            background-color: {ModernColors.SURFACE};
            border: 1px solid {ModernColors.BORDER};
            border-radius: 8px;
        }}
        
        QFrame.card {{
            background-color: {ModernColors.SURFACE};
            border: 1px solid {ModernColors.BORDER};
            border-radius: 12px;
            margin: 8px;
            padding: 16px;
        }}
        
        QFrame.panel {{
            background-color: {ModernColors.BACKGROUND};
            border: none;
            padding: 12px;
        }}
        
        QGroupBox {{
            font-weight: bold;
            border: 2px solid {ModernColors.BORDER};
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 8px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px 0 8px;
            color: {ModernColors.TEXT_PRIMARY};
            background-color: {ModernColors.SURFACE};
        }}
        """
    
    @staticmethod
    def get_lcd_style() -> str:
        """Get stylesheet for LCD displays."""
        return f"""
        QLCDNumber {{
            background-color: {ModernColors.ON_SURFACE};
            color: {ModernColors.SUCCESS};
            border: 2px solid {ModernColors.BORDER};
            border-radius: 8px;
            padding: 4px;
        }}
        
        QLCDNumber.bottle-glass {{
            color: {ModernColors.INFO};
        }}
        
        QLCDNumber.bottle-plastic {{
            color: {ModernColors.WARNING};
        }}
        
        QLCDNumber.tin-can {{
            color: {ModernColors.TEXT_SECONDARY};
        }}
        """
    
    @staticmethod
    def get_progress_style() -> str:
        """Get stylesheet for progress bars."""
        return f"""
        QProgressBar {{
            border: 2px solid {ModernColors.BORDER};
            border-radius: 8px;
            text-align: center;
            background-color: {ModernColors.BACKGROUND};
        }}
        
        QProgressBar::chunk {{
            background-color: {ModernColors.PRIMARY};
            border-radius: 6px;
        }}
        """
    
    @staticmethod
    def get_scrollbar_style() -> str:
        """Get stylesheet for scrollbars."""
        return f"""
        QScrollBar:vertical {{
            border: none;
            background: {ModernColors.BACKGROUND};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background: {ModernColors.BORDER};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: {ModernColors.TEXT_SECONDARY};
        }}
        
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        QScrollBar:horizontal {{
            border: none;
            background: {ModernColors.BACKGROUND};
            height: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background: {ModernColors.BORDER};
            border-radius: 6px;
            min-width: 20px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background: {ModernColors.TEXT_SECONDARY};
        }}
        
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {{
            border: none;
            background: none;
        }}
        """


class ThemeManager:
    """Manages application themes and styling."""
    
    def __init__(self):
        self.current_theme = "modern"
        self.themes = {
            "modern": self._get_modern_theme,
            "dark": self._get_dark_theme,
            "light": self._get_light_theme
        }
    
    def apply_theme(self, app: QApplication, theme_name: str = "modern"):
        """
        Apply theme to application.
        
        Args:
            app: QApplication instance
            theme_name: Name of theme to apply
        """
        try:
            if theme_name not in self.themes:
                logger.warning(f"Unknown theme: {theme_name}, using modern")
                theme_name = "modern"
            
            stylesheet = self.themes[theme_name]()
            app.setStyleSheet(stylesheet)
            
            self.current_theme = theme_name
            logger.info(f"Applied theme: {theme_name}")
            
        except Exception as e:
            logger.error(f"Error applying theme: {e}")
    
    def _get_modern_theme(self) -> str:
        """Get modern theme stylesheet."""
        styles = [
            ModernStyleSheet.get_main_window_style(),
            ModernStyleSheet.get_button_style(),
            ModernStyleSheet.get_input_style(),
            ModernStyleSheet.get_label_style(),
            ModernStyleSheet.get_container_style(),
            ModernStyleSheet.get_lcd_style(),
            ModernStyleSheet.get_progress_style(),
            ModernStyleSheet.get_scrollbar_style()
        ]
        
        return "\n".join(styles)
    
    def _get_dark_theme(self) -> str:
        """Get dark theme stylesheet."""
        # Override colors for dark theme
        dark_colors = {
            "BACKGROUND": "#303030",
            "SURFACE": "#424242",
            "ON_SURFACE": "#121212",
            "TEXT_PRIMARY": "#FFFFFF",
            "TEXT_SECONDARY": "#CCCCCC",
            "BORDER": "#616161"
        }
        
        # Create dark theme stylesheet
        # This would involve modifying the modern theme with dark colors
        return self._get_modern_theme()  # Simplified for now
    
    def _get_light_theme(self) -> str:
        """Get light theme stylesheet."""
        # Light theme is essentially the modern theme
        return self._get_modern_theme()
    
    def get_available_themes(self) -> list:
        """Get list of available themes."""
        return list(self.themes.keys())
    
    def get_current_theme(self) -> str:
        """Get current theme name."""
        return self.current_theme


# Global theme manager instance
theme_manager = ThemeManager()


def apply_modern_style(app: QApplication):
    """Apply modern styling to the application."""
    theme_manager.apply_theme(app, "modern")


def get_icon_color() -> str:
    """Get appropriate icon color for current theme."""
    return ModernColors.TEXT_PRIMARY


def get_status_color(status: str) -> str:
    """
    Get color for status indicators.
    
    Args:
        status: Status type ("success", "warning", "error", "info")
        
    Returns:
        Color hex string
    """
    colors = {
        "success": ModernColors.SUCCESS,
        "warning": ModernColors.WARNING,
        "error": ModernColors.ERROR,
        "info": ModernColors.INFO
    }
    
    return colors.get(status, ModernColors.TEXT_PRIMARY)