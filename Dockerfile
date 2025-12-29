# Base image
FROM python:3.11-slim

# Prevent Python buffering (better logs)
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir ultralytics opencv-python

# Default command
CMD ["python", "src/main.py", "--source", "0", "--show"]

