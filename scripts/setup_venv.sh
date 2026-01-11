#!/usr/bin/env bash
set -euo pipefail

# Create and populate a Python 3.11 virtual environment for this project.
# This script requires `python3.11` to be installed on the system.

PYTHON=${PYTHON:-python3.11}
VENV_DIR=${VENV_DIR:-.venv}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: $PYTHON not found. Install Python 3.11 and re-run this script." >&2
  exit 2
fi

echo "Creating virtual environment in $VENV_DIR using $PYTHON..."
$PYTHON -m venv "$VENV_DIR"
echo "Activating venv and upgrading pip..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo
echo "Setup complete. To activate the venv and run the app:"
echo "  source $VENV_DIR/bin/activate"
echo "  python app.py"

echo "If you need a different Python version, set PYTHON env var, e.g.:"
echo "  PYTHON=python3.10 ./scripts/setup_venv.sh"
