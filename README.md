# Signly

SBHacks XII Project - Hand gesture recognition and sign language interpretation.

Note: GPT is used heavily on this branch.

## Setup (recommended)

This project requires a Python version compatible with TensorFlow (use Python 3.11).

1. Install system Python 3.11 (if not already installed).
2. Run the helper script to create and install into a venv:

```
./scripts/setup_venv.sh
```

3. Activate the venv and run the app:

```
source .venv/bin/activate
python app.py
```

If you prefer a different venv path or Python executable, set `VENV_DIR` or `PYTHON` env vars when running the setup script.

If you run `python bootstrap.py` and see "ModuleNotFoundError" for `cv2`, `mediapipe`, `numpy`, or `flask`, make sure you created the `.venv` and re-run using the venv Python. The bootstrap script will automatically re-launch itself with `.venv/bin/python` if that file exists. To manually ensure correct environment:

```
./scripts/setup_venv.sh
source .venv/bin/activate
python bootstrap.py
```

