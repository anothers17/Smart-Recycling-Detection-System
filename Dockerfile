# Smart Recycling Detection System - Docker Configuration
# Multi-stage build for optimized production image

# Build stage
FROM python:3.10-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopencv-imgproc4.5 \
    libopencv-highgui4.5 \
    libopencv-core4.5 \
    libopencv-imgcodecs4.5 \
    libopencv-videoio4.5 \
    libglib2.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libxkbcommon-x11-0 \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create application user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p logs output && \
    chown -R app:app /app

# Switch to application user
USER app

# Set environment variables
ENV PYTHONPATH=/app
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.main; print('OK')" || exit 1

# Default command
CMD ["python", "src/main.py"]

# Labels for metadata
LABEL maintainer="sulhee8@gmail.com"
LABEL version="1.0.0"
LABEL description="Smart Recycling Detection System"
LABEL org.opencontainers.image.title="Smart Recycling Detection"
LABEL org.opencontainers.image.description="AI-powered recycling detection system using YOLOv8"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Sulhee Sama-alee <sulhee8@gmail.com>"
LABEL org.opencontainers.image.url="https://github.com/anothers17/smart-recycling-detection"
LABEL org.opencontainers.image.source="https://github.com/anothers17/smart-recycling-detection"
LABEL org.opencontainers.image.licenses="MIT"