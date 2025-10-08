#!/bin/bash

echo "🏔️ Starting Landslide Risk Prediction Application"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed"
    echo "Please install Python3 first"
    exit 1
fi

echo "✅ Python3 found"
echo

# Make the script executable
chmod +x run.py

# Run the application
python3 run.py
