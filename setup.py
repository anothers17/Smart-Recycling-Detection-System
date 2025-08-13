"""
Smart Recycling Detection System
Setup configuration for package installation.

Author: Sulhee Sama-alee
Date: 2024-
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
README = (Path(__file__).parent / "README.md").read_text()

# Read requirements
requirements = []
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="smart-recycling-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered recyclable material detection and counting system",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smart-recycling-detection",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/smart-recycling-detection/issues",
        "Documentation": "https://github.com/yourusername/smart-recycling-detection/wiki",
        "Source Code": "https://github.com/yourusername/smart-recycling-detection",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-qt>=4.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "nvidia-ml-py>=11.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "smart-recycling=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": [
            "resources/icons/*",
            "resources/models/*",
            "gui/styles/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "computer-vision",
        "object-detection",
        "recycling",
        "yolo",
        "ai",
        "machine-learning",
        "pyqt5",
        "opencv",
        "environmental",
        "sustainability",
    ],
)