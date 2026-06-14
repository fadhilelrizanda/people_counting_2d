#!/bin/bash

# Exit on error
set -e

# Define directories
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/sample_videos"
MODELS_DIR="${PROJECT_DIR}/models"
OUTPUT_DIR="${PROJECT_DIR}/output"

# Create directories if they do not exist
echo "📁 Creating directory structure..."
mkdir -p "${DATA_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Download sample video
echo "📥 Downloading sample video (pedestrian walkway)..."
curl -L -o "${DATA_DIR}/people_demo.mp4" \
  "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"

# Download pre-trained YOLOv8n model weights
echo "📥 Downloading pre-trained YOLOv8n weights..."
curl -L -o "${MODELS_DIR}/yolov8n.pt" \
  "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

echo "✅ Setup and downloads complete!"
echo "   - Demo Video: data/sample_videos/people_demo.mp4"
echo "   - Base Model: models/yolov8n.pt"
echo "   - Output Dir: output/"
