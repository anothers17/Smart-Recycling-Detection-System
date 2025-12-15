# Smart Recycling Detection System - Fixed Docker Configuration
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
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set work directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment for better dependency isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies with correct package names
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies (using available package names)
    libopencv-dev \
    python3-opencv \
    # Alternative: install specific OpenCV libraries
    # libopencv-core406 \
    # libopencv-imgproc406 \
    # libopencv-imgcodecs406 \
    # libopencv-highgui406 \
    # libopencv-videoio406 \
    # GUI dependencies (minimal set for headless operation)
    libglib2.0-0 \
    # Uncomment these if you need full GUI support
    # libgtk-3-0 \
    # libx11-xcb1 \
    # libxcb-icccm4 \
    # libxcb-image0 \
    # libxcb-keysyms1 \
    # libxcb-randr0 \
    # libxcb-render-util0 \
    # libxcb-xinerama0 \
    # libxcb-xfixes0 \
    # libxkbcommon-x11-0 \
    libfontconfig1 \
    libfreetype6 \
    # Additional useful tools
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application user with specific UID/GID for better security
RUN groupadd -r app --gid=1000 && \
    useradd -r -g app --uid=1000 --home-dir=/app --shell=/bin/bash app

# Set work directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=app:app . .

# Create necessary directories with proper permissions
RUN mkdir -p logs output models cache && \
    chown -R app:app /app && \
    chmod -R 755 /app

# Switch to application user
USER app

# Set environment variables
ENV PYTHONPATH=/app
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (uncomment if your app serves HTTP)
# EXPOSE 8000

# Enhanced health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app'); import src.main; print('Health check passed')" || exit 1

# Use exec form for better signal handling
CMD ["python", "-m", "src.main"]

# Enhanced labels for better metadata
LABEL maintainer="sulhee8@gmail.com"
LABEL version="1.0.0"
LABEL description="Smart Recycling Detection System using YOLOv8"
LABEL org.opencontainers.image.title="Smart Recycling Detection"
LABEL org.opencontainers.image.description="AI-powered recycling detection system using YOLOv8 and OpenCV"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Sulhee Sama-alee <sulhee8@gmail.com>"
LABEL org.opencontainers.image.url="https://github.com/anothers17/smart-recycling-detection"
LABEL org.opencontainers.image.source="https://github.com/anothers17/smart-recycling-detection"
LABEL org.opencontainers.image.licenses="MIT"