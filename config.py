"""
OptiGemma Configuration
Handles API key rotation, model paths, and app settings.
"""
import os
import itertools
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Gemma API Key Pool — Round-Robin Rotation
# ---------------------------------------------------------------------------
def _load_gemma_keys():
    """Load all GEMMA_API_KEY_* from .env and return as a list."""
    keys = []
    for key, value in os.environ.items():
        if key.startswith("GEMMA_API_KEY_") and value and value != "your_key_here":
            keys.append(value)
    if not keys:
        print("[WARNING] No Gemma API keys found in .env! Add at least one GEMMA_API_KEY_1=...")
    return keys

GEMMA_KEYS = _load_gemma_keys()
_key_cycle = itertools.cycle(GEMMA_KEYS) if GEMMA_KEYS else None

def get_next_gemma_key():
    """Get the next API key from the rotation pool."""
    if _key_cycle is None:
        raise RuntimeError("No Gemma API keys configured. Add keys to .env file.")
    return next(_key_cycle)

# ---------------------------------------------------------------------------
# Model Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "static", "results")

# Tanwar-12 ResNet50 model
DR_MODEL_PATH = os.path.join(MODELS_DIR, "model.h5")

# RishiSwethan vessel segmentation model
VESSEL_MODEL_DIR = os.path.join(MODELS_DIR, "vessel_model")

# ---------------------------------------------------------------------------
# App Settings
# ---------------------------------------------------------------------------
FLASK_SECRET = os.getenv("FLASK_SECRET_KEY", "optigemma-dev-key")
DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

# Image settings
IMG_SIZE = 64    # Tanwar-12 model expects 64x64
DISPLAY_SIZE = 512  # Display images at higher resolution
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

# DR Stage Labels
DR_STAGES = {
    0: {"name": "No DR", "severity": "none", "color": "#22c55e"},
    1: {"name": "Mild NPDR", "severity": "mild", "color": "#eab308"},
    2: {"name": "Moderate NPDR", "severity": "moderate", "color": "#f97316"},
    3: {"name": "Severe NPDR", "severity": "severe", "color": "#ef4444"},
    4: {"name": "Proliferative DR", "severity": "proliferative", "color": "#dc2626"},
}

# Gemma Model
GEMMA_MODEL_NAME = "gemma-4-31b-it"

# Ensure directories exist
for d in [MODELS_DIR, UPLOAD_DIR, RESULTS_DIR, VESSEL_MODEL_DIR]:
    os.makedirs(d, exist_ok=True)
