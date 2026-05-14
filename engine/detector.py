"""
OptiGemma -- DR Detection Engine (Dual-Model Architecture)
Primary: EfficientNet-B3 (PyTorch, 43MB, 5-class, 300x300)
  Supports both timm (Kaggle-trained) and torchvision (RishiSwethan) formats.
Fallback: Tanwar-12's CNN (TensorFlow, 0.4MB, regression, 64x64)
"""
import numpy as np
import os
import json
from config import DR_MODEL_PATH, DR_STAGES, IMG_SIZE

# Model cache
_pytorch_model = None
_tf_model = None
_active_model_type = None  # 'pytorch' or 'tensorflow'

# Paths
PYTORCH_MODEL_PATH = os.path.join(os.path.dirname(DR_MODEL_PATH), 'vessel_model', 'best_val_loss.pt')
PYTORCH_HP_PATH = os.path.join(os.path.dirname(DR_MODEL_PATH), 'vessel_model', 'best_hp.json')

# EfficientNet-B3 input size
EFFNET_INPUT_SIZE = 300


def _load_pytorch_model():
    """Load EfficientNet-B3 model. Supports both timm and torchvision formats."""
    global _pytorch_model, _active_model_type

    if _pytorch_model is not None:
        _active_model_type = 'pytorch'
        return _pytorch_model

    if not os.path.exists(PYTORCH_MODEL_PATH):
        print("[WARNING] EfficientNet-B3 model not found at {}".format(PYTORCH_MODEL_PATH))
        return None

    try:
        import torch

        print("[MODEL] Loading EfficientNet-B3 from {}...".format(PYTORCH_MODEL_PATH))
        state_dict = torch.load(PYTORCH_MODEL_PATH, map_location='cpu', weights_only=False)

        # Handle checkpoint wrapper (from train_model.py)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        # Remove 'model.' prefix if present (RishiSwethan format)
        clean_sd = {}
        for key, value in state_dict.items():
            clean_sd[key[6:] if key.startswith('model.') else key] = value

        # Detect format: timm uses 'conv_stem.weight', torchvision uses 'features.0.0.weight'
        is_timm = 'conv_stem.weight' in clean_sd

        if is_timm:
            # New Kaggle-trained model (timm format)
            try:
                import timm
            except ImportError:
                print("[WARNING] timm not installed, trying pip install...")
                os.system('pip install -q timm')
                import timm
            model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=5)
            model.load_state_dict(clean_sd, strict=True)
            print("[OK] EfficientNet-B3 loaded (timm/Kaggle format, 5-class, 300x300)")
        else:
            # Old RishiSwethan model (torchvision format)
            import torchvision.models as models
            model = models.efficientnet_b3(weights=None)
            num_features = model.classifier[1].in_features  # 1536

            # Auto-detect classifier architecture
            has_enhanced_head = any('classifier.3' in k for k in clean_sd)
            if has_enhanced_head:
                model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.4),
                    torch.nn.Linear(num_features, 512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.Dropout(p=0.3),
                    torch.nn.Linear(512, 5),
                )
                print("[MODEL] Detected enhanced classifier head")
            else:
                model.classifier = torch.nn.Linear(num_features, 5)

            model.load_state_dict(clean_sd, strict=True)
            print("[OK] EfficientNet-B3 loaded (torchvision format, 5-class, 300x300)")

        model.eval()
        _pytorch_model = model
        _active_model_type = 'pytorch'
        return model

    except Exception as e:
        print("[WARNING] Failed to load EfficientNet-B3: {}".format(e))
        print("[FALLBACK] Will try Tanwar-12 TF model...")
        return None


def _load_tf_model():
    """Load Tanwar-12's TensorFlow CNN model (fallback)."""
    global _tf_model, _active_model_type

    if _tf_model is not None:
        _active_model_type = 'tensorflow'
        return _tf_model

    if not os.path.exists(DR_MODEL_PATH):
        print("[WARNING] TF Model not found at {}".format(DR_MODEL_PATH))
        return None

    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        print("[MODEL] Loading Tanwar-12 CNN from {}...".format(DR_MODEL_PATH))
        _tf_model = tf.keras.models.load_model(DR_MODEL_PATH, compile=False)
        _active_model_type = 'tensorflow'
        print("[OK] TF Model loaded! Input shape: {}".format(_tf_model.input_shape))
        return _tf_model
    except Exception as e:
        print("[ERROR] Failed to load TF model: {}".format(e))
        return None


def _load_model():
    """Load the best available model. Tries EfficientNet-B3 first, then TF CNN."""
    # Try PyTorch EfficientNet-B3 first (better accuracy)
    model = _load_pytorch_model()
    if model is not None:
        return model

    # Fallback to TensorFlow CNN
    model = _load_tf_model()
    if model is not None:
        return model

    print("[ERROR] No model available!")
    return None


def _preprocess_for_effnet(image):
    """
    Preprocess image for EfficientNet-B3 inference.
    Input: numpy array of any shape (H, W, 3), uint8 or float
    Output: (300, 300, 3) normalized numpy array
    """
    import cv2

    # Ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Resize to 300x300 for EfficientNet-B3
    resized = cv2.resize(image, (EFFNET_INPUT_SIZE, EFFNET_INPUT_SIZE))

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def predict(preprocessed_image):
    """
    Run DR classification on a preprocessed image.

    Args:
        preprocessed_image: numpy array, normalized to [0, 1]

    Returns:
        dict with stage, stage_name, confidence, all_probabilities, etc.
    """
    model = _load_model()

    if model is None:
        return _mock_prediction()

    if _active_model_type == 'pytorch':
        return _predict_pytorch(preprocessed_image, model)
    else:
        return _predict_tensorflow(preprocessed_image, model)


def _predict_pytorch(image, model):
    """Run prediction with EfficientNet-B3 PyTorch model."""
    import torch

    # Preprocess for EfficientNet
    effnet_input = _preprocess_for_effnet(image)

    # Convert to PyTorch tensor: (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(effnet_input).permute(2, 0, 1).unsqueeze(0)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    # Inference
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1).numpy()[0]

    stage = int(np.argmax(probs))
    confidence = float(probs[stage] * 100)
    all_probs = {i: float(probs[i] * 100) for i in range(5)}

    stage_info = DR_STAGES.get(stage, DR_STAGES[0])

    return {
        "stage": stage,
        "stage_name": stage_info["name"],
        "confidence": round(confidence, 1),
        "all_probabilities": all_probs,
        "severity": stage_info["severity"],
        "color": stage_info["color"],
        "_model": "EfficientNet-B3 (OptiGemma)",
    }


def _predict_tensorflow(image, model):
    """Run prediction with Tanwar-12's TF CNN model."""
    # Expand dims for batch
    input_batch = np.expand_dims(image, axis=0)

    predictions = model.predict(input_batch, verbose=0)

    if predictions.shape[-1] == 1:
        # Regression output
        stage = int(np.clip(np.round(predictions[0][0]), 0, 4))
        confidence = max(60.0, 95.0 - abs(predictions[0][0] - stage) * 30)
        all_probs = {i: (90.0 if i == stage else 2.5) for i in range(5)}
    else:
        # Classification output
        probs = predictions[0]
        if np.min(probs) < 0 or np.sum(probs) > 1.5:
            exp_probs = np.exp(probs - np.max(probs))
            probs = exp_probs / exp_probs.sum()
        stage = int(np.argmax(probs))
        confidence = float(probs[stage] * 100)
        all_probs = {i: float(probs[i] * 100) for i in range(5)}

    stage_info = DR_STAGES.get(stage, DR_STAGES[0])

    return {
        "stage": stage,
        "stage_name": stage_info["name"],
        "confidence": round(confidence, 1),
        "all_probabilities": all_probs,
        "severity": stage_info["severity"],
        "color": stage_info["color"],
        "_model": "CNN (Tanwar-12)",
    }


def _mock_prediction():
    """Return a mock prediction for development when no model is available."""
    import random
    stage = random.choice([0, 1, 2, 2, 3])
    confidence = round(random.uniform(75, 98), 1)
    stage_info = DR_STAGES[stage]
    return {
        "stage": stage,
        "stage_name": stage_info["name"],
        "confidence": confidence,
        "all_probabilities": {
            0: round(random.uniform(1, 10), 1) if stage != 0 else confidence,
            1: round(random.uniform(1, 10), 1) if stage != 1 else confidence,
            2: round(random.uniform(1, 10), 1) if stage != 2 else confidence,
            3: round(random.uniform(1, 10), 1) if stage != 3 else confidence,
            4: round(random.uniform(1, 5), 1) if stage != 4 else confidence,
        },
        "severity": stage_info["severity"],
        "color": stage_info["color"],
        "_mock": True,
    }


def get_model_for_gradcam():
    """Return the loaded Keras model (for Grad-CAM). Falls back to TF model."""
    # Grad-CAM works with TF/Keras, so always return TF model for that
    return _load_tf_model()
