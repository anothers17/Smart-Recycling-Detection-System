"""
File handling utilities for Smart Recycling Detection System.

This module provides utilities for file operations including
validation, path management, and file type checking.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import logging
from datetime import datetime

from config.logging_config import get_logger

logger = get_logger('main')


# Supported file formats
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
SUPPORTED_MODEL_FORMATS = {'.pt', '.pth', '.onnx', '.trt', '.engine'}


def validate_file_path(filepath: Union[str, Path], check_exists: bool = True) -> bool:
    """
    Validate if a file path is valid and optionally exists.
    
    Args:
        filepath: Path to validate
        check_exists: Whether to check if file exists
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(filepath)
        
        if check_exists and not path.exists():
            logger.warning(f"File does not exist: {filepath}")
            return False
        
        if check_exists and not path.is_file():
            logger.warning(f"Path is not a file: {filepath}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating file path {filepath}: {e}")
        return False


def validate_directory_path(dirpath: Union[str, Path], create_if_missing: bool = False) -> bool:
    """
    Validate if a directory path is valid and optionally create it.
    
    Args:
        dirpath: Directory path to validate
        create_if_missing: Whether to create directory if it doesn't exist
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(dirpath)
        
        if not path.exists():
            if create_if_missing:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dirpath}")
            else:
                logger.warning(f"Directory does not exist: {dirpath}")
                return False
        
        if not path.is_dir():
            logger.warning(f"Path is not a directory: {dirpath}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating directory path {dirpath}: {e}")
        return False


def is_image_file(filepath: Union[str, Path]) -> bool:
    """
    Check if file is a supported image format.
    
    Args:
        filepath: Path to check
        
    Returns:
        True if supported image format, False otherwise
    """
    try:
        path = Path(filepath)
        return path.suffix.lower() in SUPPORTED_IMAGE_FORMATS
    except Exception:
        return False


def is_video_file(filepath: Union[str, Path]) -> bool:
    """
    Check if file is a supported video format.
    
    Args:
        filepath: Path to check
        
    Returns:
        True if supported video format, False otherwise
    """
    try:
        path = Path(filepath)
        return path.suffix.lower() in SUPPORTED_VIDEO_FORMATS
    except Exception:
        return False


def is_model_file(filepath: Union[str, Path]) -> bool:
    """
    Check if file is a supported model format.
    
    Args:
        filepath: Path to check
        
    Returns:
        True if supported model format, False otherwise
    """
    try:
        path = Path(filepath)
        return path.suffix.lower() in SUPPORTED_MODEL_FORMATS
    except Exception:
        return False


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        filepath: Path to analyze
        
    Returns:
        Dictionary with file information
    """
    try:
        path = Path(filepath)
        
        if not path.exists():
            return {'error': 'File does not exist'}
        
        stat = path.stat()
        
        info = {
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'parent': str(path.parent),
            'absolute_path': str(path.absolute()),
            'is_image': is_image_file(path),
            'is_video': is_video_file(path),
            'is_model': is_model_file(path)
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting file info for {filepath}: {e}")
        return {'error': str(e)}


def find_files(directory: Union[str, Path], pattern: str = "*", 
               recursive: bool = True, file_types: Optional[List[str]] = None) -> List[Path]:
    """
    Find files in directory matching criteria.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        recursive: Whether to search recursively
        file_types: List of file extensions to filter by
        
    Returns:
        List of matching file paths
    """
    try:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
        
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))
        
        # Filter by file types if specified
        if file_types:
            file_types = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                         for ext in file_types]
            files = [f for f in files if f.suffix.lower() in file_types]
        
        # Filter to only files (not directories)
        files = [f for f in files if f.is_file()]
        
        logger.debug(f"Found {len(files)} files in {directory}")
        return sorted(files)
        
    except Exception as e:
        logger.error(f"Error finding files in {directory}: {e}")
        return []


def find_model_files(directory: Union[str, Path]) -> List[Path]:
    """
    Find all model files in directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of model file paths
    """
    return find_files(directory, file_types=list(SUPPORTED_MODEL_FORMATS))


def find_video_files(directory: Union[str, Path]) -> List[Path]:
    """
    Find all video files in directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of video file paths
    """
    return find_files(directory, file_types=list(SUPPORTED_VIDEO_FORMATS))


def find_image_files(directory: Union[str, Path]) -> List[Path]:
    """
    Find all image files in directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of image file paths
    """
    return find_files(directory, file_types=list(SUPPORTED_IMAGE_FORMATS))


def safe_copy_file(src: Union[str, Path], dst: Union[str, Path], 
                   overwrite: bool = False) -> bool:
    """
    Safely copy a file with error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            logger.error(f"Source file does not exist: {src}")
            return False
        
        if dst_path.exists() and not overwrite:
            logger.warning(f"Destination file exists and overwrite=False: {dst}")
            return False
        
        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src_path, dst_path)
        logger.info(f"File copied: {src} -> {dst}")
        return True
        
    except Exception as e:
        logger.error(f"Error copying file {src} to {dst}: {e}")
        return False


def safe_move_file(src: Union[str, Path], dst: Union[str, Path], 
                   overwrite: bool = False) -> bool:
    """
    Safely move a file with error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            logger.error(f"Source file does not exist: {src}")
            return False
        
        if dst_path.exists() and not overwrite:
            logger.warning(f"Destination file exists and overwrite=False: {dst}")
            return False
        
        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(src_path), str(dst_path))
        logger.info(f"File moved: {src} -> {dst}")
        return True
        
    except Exception as e:
        logger.error(f"Error moving file {src} to {dst}: {e}")
        return False


def safe_delete_file(filepath: Union[str, Path]) -> bool:
    """
    Safely delete a file with error handling.
    
    Args:
        filepath: Path to file to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(filepath)
        
        if not path.exists():
            logger.warning(f"File does not exist: {filepath}")
            return True  # Already deleted
        
        path.unlink()
        logger.info(f"File deleted: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting file {filepath}: {e}")
        return False


def create_backup(filepath: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Create a backup of a file.
    
    Args:
        filepath: Path to file to backup
        backup_dir: Directory for backup (default: same directory)
        
    Returns:
        Path to backup file or None if failed
    """
    try:
        src_path = Path(filepath)
        
        if not src_path.exists():
            logger.error(f"Source file does not exist: {filepath}")
            return None
        
        if backup_dir is None:
            backup_dir = src_path.parent
        else:
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{src_path.stem}_backup_{timestamp}{src_path.suffix}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(src_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"Error creating backup for {filepath}: {e}")
        return None


def load_json(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load JSON data from file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        path = Path(filepath)
        
        if not path.exists():
            logger.warning(f"JSON file does not exist: {filepath}")
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"JSON loaded: {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        return None


def save_json(data: Dict[str, Any], filepath: Union[str, Path], 
              indent: int = 2, create_backup: bool = True) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save to
        indent: JSON indentation
        create_backup: Whether to create backup if file exists
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(filepath)
        
        # Create backup if file exists
        if create_backup and path.exists():
            create_backup(path)
        
        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        logger.info(f"JSON saved: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        return False


def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Get total size of directory in bytes.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    try:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return 0
        
        total_size = 0
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
        
    except Exception as e:
        logger.error(f"Error calculating directory size for {directory}: {e}")
        return 0


def cleanup_old_files(directory: Union[str, Path], max_age_days: int, 
                     pattern: str = "*", dry_run: bool = False) -> List[Path]:
    """
    Clean up old files in directory.
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum age in days
        pattern: File pattern to match
        dry_run: If True, only return files that would be deleted
        
    Returns:
        List of files that were (or would be) deleted
    """
    try:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return []
        
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        old_files = []
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                file_mtime = file_path.stat().st_mtime
                if file_mtime < cutoff_time:
                    old_files.append(file_path)
                    if not dry_run:
                        safe_delete_file(file_path)
        
        if dry_run:
            logger.info(f"Would delete {len(old_files)} old files from {directory}")
        else:
            logger.info(f"Deleted {len(old_files)} old files from {directory}")
        
        return old_files
        
    except Exception as e:
        logger.error(f"Error cleaning up old files in {directory}: {e}")
        return []


def ensure_unique_filename(filepath: Union[str, Path]) -> Path:
    """
    Ensure filename is unique by adding counter if needed.
    
    Args:
        filepath: Desired file path
        
    Returns:
        Unique file path
    """
    try:
        path = Path(filepath)
        
        if not path.exists():
            return path
        
        counter = 1
        while True:
            new_name = f"{path.stem}_{counter}{path.suffix}"
            new_path = path.parent / new_name
            
            if not new_path.exists():
                return new_path
            
            counter += 1
            
            # Safety limit
            if counter > 1000:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_name = f"{path.stem}_{timestamp}{path.suffix}"
                return path.parent / final_name
        
    except Exception as e:
        logger.error(f"Error ensuring unique filename for {filepath}: {e}")
        return Path(filepath)


class FileWatcher:
    """Simple file watcher for monitoring file changes."""
    
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.last_modified = None
        self._update_modified_time()
    
    def _update_modified_time(self):
        """Update the last modified time."""
        try:
            if self.filepath.exists():
                self.last_modified = self.filepath.stat().st_mtime
        except Exception:
            self.last_modified = None
    
    def has_changed(self) -> bool:
        """Check if file has been modified since last check."""
        try:
            if not self.filepath.exists():
                return self.last_modified is not None
            
            current_modified = self.filepath.stat().st_mtime
            changed = current_modified != self.last_modified
            
            if changed:
                self.last_modified = current_modified
            
            return changed
            
        except Exception:
            return False
    
    def reset(self):
        """Reset the watcher."""
        self._update_modified_time()