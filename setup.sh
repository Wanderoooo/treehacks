#!/bin/bash
# Setup script for Bike Detection and Analysis System

echo "=================================================="
echo "Bike Detection and Analysis System - Setup"
echo "=================================================="
echo

# Check Python version
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

# Create virtual environment
echo
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if Ollama is installed
echo
echo "Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"

    # Check if LLaVA model is available
    echo "Checking for LLaVA model..."
    if ollama list | grep -q "llava"; then
        echo "✓ LLaVA model is available"
    else
        echo "LLaVA model not found. Pulling llava:latest..."
        ollama pull llava:latest
    fi
else
    echo "⚠ Ollama is not installed"
    echo "Please install Ollama from https://ollama.ai/download"
    echo "After installation, run: ollama pull llava:latest"
fi

# Create necessary directories
echo
echo "Creating directories..."
mkdir -p input output/videos output/reports

# Test YOLO model download
echo
echo "Testing YOLO model download..."
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" 2>&1 | grep -q "Downloading" && echo "✓ YOLO model downloaded" || echo "✓ YOLO model ready"

echo
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo
echo "Next steps:"
echo "1. Place your video files in the 'input/' directory"
echo "2. Run: python main.py input/your_video.mp4"
echo
echo "For help: python main.py --help"
echo
