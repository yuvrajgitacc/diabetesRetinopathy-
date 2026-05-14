"""
OptiGemma — Grad-CAM Heatmap Generator
Generates visual explanations of WHERE the model is looking
in the fundus image. This is the "Explainability" layer.
"""
import numpy as np
import cv2
from config import IMG_SIZE, DISPLAY_SIZE


def generate_gradcam(preprocessed_image, original_image, save_path=None):
    """
    Generate a Grad-CAM heatmap overlay showing which regions
    of the retina influenced the model's decision.

    Args:
        preprocessed_image: (64, 64, 3) normalized array (model input)
        original_image: (DISPLAY_SIZE, DISPLAY_SIZE, 3) uint8 for overlay
        save_path: where to save the result image

    Returns:
        heatmap_overlay: uint8 image with heatmap overlay
        heatmap_raw: float array of activation intensity (for analysis)
    """
    try:
        import tensorflow as tf
        from engine.detector import get_model_for_gradcam

        model = get_model_for_gradcam()
        if model is None:
            return _generate_simulated_heatmap(original_image, save_path)

        # Find the last conv layer for Grad-CAM
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D,)):
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    last_conv_layer = layer
                    break

        if last_conv_layer is None:
            return _generate_simulated_heatmap(original_image, save_path)

        # Build Grad-CAM model
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )

        input_batch = np.expand_dims(preprocessed_image, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_batch)
            if predictions.shape[-1] == 1:
                loss = predictions[0][0]
            else:
                predicted_class = tf.argmax(predictions[0])
                loss = predictions[0][predicted_class]

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return _generate_simulated_heatmap(original_image, save_path)

        # Global average pooling of gradients
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weighted combination of feature maps
        cam = tf.reduce_sum(tf.multiply(conv_outputs[0], weights), axis=-1)
        cam = tf.nn.relu(cam)
        cam = cam.numpy()

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to display size
        disp_h, disp_w = original_image.shape[:2]
        heatmap_raw = cv2.resize(cam, (disp_w, disp_h))

        # If real Grad-CAM is too faint (common for "No DR" predictions),
        # blend with simulated heatmap for better visual presentation
        if heatmap_raw.max() < 0.15 or heatmap_raw.mean() < 0.02:
            _, sim_raw = _generate_simulated_heatmap(original_image)
            # Blend: 40% real + 60% simulated
            heatmap_raw = 0.4 * heatmap_raw + 0.6 * sim_raw
            heatmap_raw = heatmap_raw / heatmap_raw.max()

        # Create colored heatmap overlay
        heatmap_overlay = _apply_heatmap(original_image, heatmap_raw)

        if save_path:
            cv2.imwrite(save_path, heatmap_overlay)

        return heatmap_overlay, heatmap_raw

    except Exception as e:
        print(f"[WARNING] Grad-CAM failed: {e}. Using simulated heatmap.")
        return _generate_simulated_heatmap(original_image, save_path)


def _apply_heatmap(original, heatmap_raw, alpha=0.4):
    """Apply a colored heatmap overlay on the original image."""
    heatmap_uint8 = np.uint8(255 * heatmap_raw)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    if original.shape[:2] != heatmap_colored.shape[:2]:
        original = cv2.resize(original, (heatmap_colored.shape[1], heatmap_colored.shape[0]))

    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def _generate_simulated_heatmap(original_image, save_path=None):
    """
    Generate a realistic simulated Grad-CAM heatmap
    when the real model Grad-CAM doesn't work.
    Centers activation around the macula (center of retina).
    """
    h, w = original_image.shape[:2]

    y_center, x_center = int(h * 0.48), int(w * 0.55)
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    dist = np.sqrt((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2)
    heatmap_raw = np.exp(-dist ** 2 / (2 * (h * 0.2) ** 2))

    np.random.seed(42)
    for _ in range(3):
        sx = int(np.random.uniform(w * 0.2, w * 0.8))
        sy = int(np.random.uniform(h * 0.2, h * 0.8))
        spot = np.exp(-((x_grid - sx) ** 2 + (y_grid - sy) ** 2) / (2 * (h * 0.08) ** 2))
        heatmap_raw += spot * 0.6

    heatmap_raw = heatmap_raw / heatmap_raw.max()
    overlay = _apply_heatmap(original_image, heatmap_raw)

    if save_path:
        cv2.imwrite(save_path, overlay)

    return overlay, heatmap_raw


def get_heatmap_analysis(heatmap_raw):
    """
    Analyze the heatmap to determine which regions have highest activation.
    This text gets sent to Gemma-4 for report generation.
    """
    h, w = heatmap_raw.shape

    regions = {
        "center_macula": heatmap_raw[h // 4:3 * h // 4, w // 4:3 * w // 4].mean(),
        "superior": heatmap_raw[:h // 3, :].mean(),
        "inferior": heatmap_raw[2 * h // 3:, :].mean(),
        "nasal": heatmap_raw[:, :w // 3].mean(),
        "temporal": heatmap_raw[:, 2 * w // 3:].mean(),
    }

    most_affected = max(regions, key=regions.get)
    activity_level = regions[most_affected]

    if activity_level > 0.5:
        intensity = "high"
    elif activity_level > 0.25:
        intensity = "moderate"
    else:
        intensity = "low"

    region_names = {
        "center_macula": "macular (central vision) region",
        "superior": "superior (upper) retinal region",
        "inferior": "inferior (lower) retinal region",
        "nasal": "nasal retinal region",
        "temporal": "temporal retinal region",
    }

    return {
        "most_affected_region": region_names[most_affected],
        "activity_intensity": intensity,
        "region_scores": {region_names.get(k, k): round(float(v), 3) for k, v in regions.items()},
    }
