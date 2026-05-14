"""
OptiGemma — Image Preprocessor
Inspired by Bhimrazy + Tanwar-12 preprocessing pipelines.

Performs:
  1. Circular crop (remove black borders from fundus images)
  2. Gaussian blur (enhance vessel visibility)
  3. Ben Graham's color normalization
  4. Resize to model input size (64x64) AND display size (512x512)
"""
import cv2
import numpy as np
from config import IMG_SIZE, DISPLAY_SIZE


def circular_crop(img):
    """
    Crop the circular fundus region from the image, removing black borders.
    This is critical because fundus cameras produce circular images on
    rectangular sensors — the black border confuses the model.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold to find the bright fundus region
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Find the largest contour (the fundus circle)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # Return original if no contour found

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add small padding
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)

    cropped = img[y:y + h, x:x + w]
    return cropped


def apply_gaussian_blur(img, sigmaX=10):
    """
    Apply Gaussian blur to enhance vessel visibility.
    This is the Bhimrazy/Tanwar technique — a weighted blend of the
    original and blurred image brings out micro-features.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX)
    # Ben Graham's preprocessing: img = img - blurred + 128
    enhanced = cv2.addWeighted(img, 4, blurred, -4, 128)
    return enhanced


def preprocess_image(image_path, apply_enhancement=True):
    """
    Full preprocessing pipeline for a fundus image.

    Args:
        image_path: Path to the input image
        apply_enhancement: Whether to apply Gaussian enhancement

    Returns:
        processed: numpy array ready for model input (64, 64, 3)
        original_resized: clean resized original for display
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Step 1: Circular crop
    cropped = circular_crop(img)

    # Keep a clean copy for display
    original_resized = cv2.resize(cropped, (DISPLAY_SIZE, DISPLAY_SIZE))

    # Step 2: Gaussian enhancement (for model input)
    if apply_enhancement:
        enhanced = apply_gaussian_blur(cropped)
    else:
        enhanced = cropped

    # Step 3: Resize to model input size
    resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))

    # Step 4: Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized, original_resized


def preprocess_for_display(image_path):
    """
    Preprocess image and return both the model-ready array
    and intermediate images for the dashboard display.

    Returns dict with:
        - model_input: (64, 64, 3) normalized array for CNN
        - original: (512, 512, 3) original resized for display
        - enhanced: (512, 512, 3) enhanced image for display
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    cropped = circular_crop(img)
    original = cv2.resize(cropped, (DISPLAY_SIZE, DISPLAY_SIZE))
    enhanced_full = apply_gaussian_blur(cropped)
    enhanced_display = cv2.resize(enhanced_full, (DISPLAY_SIZE, DISPLAY_SIZE))

    # Model input at 64x64 (enhanced, for TF CNN)
    model_resized = cv2.resize(enhanced_full, (IMG_SIZE, IMG_SIZE))
    model_input = model_resized.astype(np.float32) / 255.0

    # Raw model input (clean, no enhancement, for EfficientNet-B3)
    # Pass high-res (512x512) and let the engine downscale as needed
    model_input_raw = original.astype(np.float32) / 255.0
    
    # Enhanced high-res model input for retrained EfficientNet-B3
    model_input_enhanced_highres = enhanced_full.astype(np.float32) / 255.0

    return {
        "model_input": model_input,          # Enhanced (for TF CNN / Grad-CAM)
        "model_input_raw": model_input_raw,   # Clean (for EfficientNet-B3)
        "model_input_enhanced_highres": model_input_enhanced_highres,
        "original": original,
        "enhanced": enhanced_display,
    }
