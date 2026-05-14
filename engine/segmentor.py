"""
OptiGemma — Vessel Segmentation Engine
Wrapper around RishiSwethan's vessel segmentation logic.

If the full PyTorch segmentation model is not available,
falls back to a classical image processing approach using
Green Channel + CLAHE + morphological operations.
"""
import cv2
import numpy as np
import os


def segment_vessels(image_path_or_array, save_path=None):
    """
    Segment blood vessels from a fundus image.

    Tries the deep learning model first. If unavailable,
    falls back to classical Green Channel extraction.

    Args:
        image_path_or_array: file path or numpy array (BGR)
        save_path: optional path to save the vessel map

    Returns:
        vessel_map: (H, W, 3) uint8 image — white vessels on black bg
        vessel_stats: dict with vessel density metrics
    """
    # Load image
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array.copy()

    if img is None:
        raise ValueError("Could not load image for vessel segmentation")

    # Resize for consistent processing
    img = cv2.resize(img, (512, 512))

    # Try deep learning model first
    vessel_map = _try_deep_segmentation(img)

    if vessel_map is None:
        # Fallback: Classical approach (Green Channel + CLAHE)
        vessel_map = _classical_segmentation(img)

    # Calculate vessel statistics
    vessel_stats = _analyze_vessels(vessel_map)

    # Convert to 3-channel for display
    if len(vessel_map.shape) == 2:
        vessel_display = cv2.cvtColor(vessel_map, cv2.COLOR_GRAY2BGR)
    else:
        vessel_display = vessel_map

    # Resize to standard display size
    vessel_display = cv2.resize(vessel_display, (224, 224))

    if save_path:
        cv2.imwrite(save_path, vessel_display)

    return vessel_display, vessel_stats


def _try_deep_segmentation(img):
    """
    Attempt to use RishiSwethan's trained PyTorch model.
    Returns None if the model is not available.
    """
    from config import VESSEL_MODEL_DIR

    model_files = []
    if os.path.exists(VESSEL_MODEL_DIR):
        model_files = [f for f in os.listdir(VESSEL_MODEL_DIR)
                       if f.endswith(('.pth', '.pt', '.h5'))]

    if not model_files:
        return None

    try:
        import torch
        import torchvision.transforms as transforms

        model_path = os.path.join(VESSEL_MODEL_DIR, model_files[0])
        print(f"[VESSEL] Loading vessel segmentation model: {model_path}")

        model = torch.load(model_path, map_location='cpu', weights_only=False)
        model.eval()

        # Preprocess for PyTorch
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        # Convert output to binary mask
        if isinstance(output, dict):
            output = output.get('out', list(output.values())[0])
        mask = output.squeeze().numpy()
        if mask.ndim > 2:
            mask = mask[0]
        mask = (mask > 0.5).astype(np.uint8) * 255

        return mask

    except Exception as e:
        print(f"[WARNING] Deep segmentation failed: {e}. Using classical method.")
        return None


def _classical_segmentation(img):
    """
    Classical vessel segmentation using Green Channel extraction.

    The green channel of a fundus image has the best contrast
    for blood vessels. Combined with CLAHE and morphological ops,
    this produces a clean vessel map.
    """
    # Step 1: Extract green channel (best vessel contrast)
    green = img[:, :, 1]

    # Step 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # Step 3: Create circular mask to ignore border
    h, w = enhanced.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2 - 10, 255, -1)

    # Step 4: Morphological operations to extract vessels
    # Use a line kernel to detect elongated structures (vessels)
    kernel_sizes = [15, 17, 19]
    vessel_sum = np.zeros_like(enhanced, dtype=np.float64)

    for angle in range(0, 180, 15):
        for ksize in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, 1))
            # Rotate kernel
            center = (ksize // 2, 0)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel_rot = cv2.warpAffine(kernel, rot_matrix, (ksize, ksize))

            # Morphological black-hat (finds dark structures on light bg)
            blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel_rot)
            vessel_sum += blackhat.astype(np.float64)

    # Normalize
    vessel_sum = (vessel_sum - vessel_sum.min()) / (vessel_sum.max() - vessel_sum.min() + 1e-8)
    vessel_uint8 = (vessel_sum * 255).astype(np.uint8)

    # Step 5: Adaptive threshold
    binary = cv2.adaptiveThreshold(
        vessel_uint8, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, -3
    )

    # Apply circular mask
    binary = cv2.bitwise_and(binary, mask)

    # Step 6: Clean up with morphological operations
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_clean)

    # Remove small noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    min_area = 30
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == i] = 0

    return cleaned


def _analyze_vessels(vessel_map):
    """
    Analyze the segmented vessel map for clinical metrics.
    These stats get sent to Gemma-4 for the report.
    """
    if len(vessel_map.shape) == 3:
        binary = cv2.cvtColor(vessel_map, cv2.COLOR_BGR2GRAY)
    else:
        binary = vessel_map

    total_pixels = binary.size
    vessel_pixels = np.count_nonzero(binary)
    vessel_density = vessel_pixels / total_pixels * 100

    # Analyze vessel distribution (quadrant analysis)
    h, w = binary.shape
    quadrants = {
        "superior_temporal": binary[:h // 2, w // 2:],
        "superior_nasal": binary[:h // 2, :w // 2],
        "inferior_temporal": binary[h // 2:, w // 2:],
        "inferior_nasal": binary[h // 2:, :w // 2],
    }

    quadrant_density = {}
    for name, quad in quadrants.items():
        qd = np.count_nonzero(quad) / quad.size * 100
        quadrant_density[name] = round(qd, 2)

    # Vessel health assessment
    if vessel_density > 8:
        health = "dense_vasculature"
        health_text = "Dense vascular network observed — may indicate neovascularization"
    elif vessel_density > 4:
        health = "normal"
        health_text = "Normal vascular density"
    elif vessel_density > 2:
        health = "reduced"
        health_text = "Reduced vascular density — possible vessel dropout"
    else:
        health = "sparse"
        health_text = "Significantly sparse vasculature — indicates advanced damage"

    return {
        "vessel_density_percent": round(vessel_density, 2),
        "vessel_health": health,
        "vessel_health_text": health_text,
        "quadrant_density": quadrant_density,
    }
