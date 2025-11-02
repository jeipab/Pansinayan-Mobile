#!/bin/bash

echo "=========================================="
echo "Starting Pansinayan Server"
echo "=========================================="

# Check if models exist
if [ ! -f "SignTransformerCtc_best.pt" ]; then
    echo "ERROR: SignTransformerCtc_best.pt not found!"
    echo "Please copy your model files to the server directory."
    exit 1
fi

if [ ! -f "MediaPipeGRUCtc_best.pt" ]; then
    echo "ERROR: MediaPipeGRUCtc_best.pt not found!"
    echo "Please copy your model files to the server directory."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Loaded environment variables from .env"
else
    echo "WARNING: .env file not found, using defaults"
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if CUDA is available
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Start server
echo "=========================================="
echo "Starting Uvicorn server on ${HOST:-0.0.0.0}:${PORT:-8000}"
echo "=========================================="

python3 app.py