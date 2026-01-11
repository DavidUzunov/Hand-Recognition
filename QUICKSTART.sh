#!/bin/bash
# Quick start guide for the ASL Recognition System

echo "=========================================="
echo "Signly - ASL Recognition System"
echo "=========================================="
echo ""
echo "This script will help you get started with the hand recognition ML pipeline."
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
echo "Checking environment..."
if ! command_exists python3; then
    echo "✗ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION found"

# Check venv
if [ ! -d ".venv" ]; then
    echo "✗ Virtual environment not found"
    echo "  Creating venv..."
    python3 -m venv .venv
    ./.venv/bin/pip install -U pip setuptools wheel -q
fi
echo "✓ Virtual environment ready"

# Check dependencies
echo ""
echo "Checking dependencies..."
./.venv/bin/pip list | grep -q "Flask" && echo "✓ Flask" || echo "✗ Flask"
./.venv/bin/pip list | grep -q "tensorflow" && echo "✓ TensorFlow" || echo "✗ TensorFlow"
./.venv/bin/pip list | grep -q "opencv" && echo "✓ OpenCV" || echo "✗ OpenCV"
./.venv/bin/pip list | grep -q "mediapipe" && echo "✓ MediaPipe" || echo "✗ MediaPipe"

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. CREATE TEST DATA (5 min):"
echo "   ./.venv/bin/python train.py --create-dummy --epochs 5"
echo ""
echo "2. TRAIN MODEL (10 min):"
echo "   ./.venv/bin/python train.py --epochs 50 --batch-size 16"
echo ""
echo "3. RUN APPLICATION:"
echo "   ./.venv/bin/python bootstrap.py"
echo "   (or: python bootstrap.py - will auto-activate venv)"
echo ""
echo "4. OPEN BROWSER:"
echo "   http://localhost:5000"
echo ""
echo "=========================================="
echo "TRAINING YOUR OWN DATA:"
echo "=========================================="
echo ""
echo "A. FROM VIDEO FILES:"
echo "   ./.venv/bin/python prepare_data.py \\"
echo "       --input-videos /path/to/gesture/videos \\"
echo "       --output-dir ./data \\"
echo "       --num-frames 30"
echo ""
echo "B. FROM IMAGE SEQUENCES:"
echo "   ./.venv/bin/python prepare_data.py \\"
echo "       --input-images /path/to/frame/sequences \\"
echo "       --output-dir ./data"
echo ""
echo "C. THEN TRAIN:"
echo "   ./.venv/bin/python train.py --epochs 50"
echo ""
echo "=========================================="
echo "DOCUMENTATION:"
echo "=========================================="
echo ""
echo "• ML_PIPELINE.md - Complete ML pipeline guide"
echo "• ML_IMPLEMENTATION.md - Implementation details"
echo "• data/README.md - Data format and preparation"
echo "• README.md - Project overview"
echo ""
echo "=========================================="
