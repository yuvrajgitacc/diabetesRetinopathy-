import torch
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter
import cv2

from engine.detector import _load_pytorch_model, _load_tf_model, _predict_pytorch, _predict_tensorflow
from engine.preprocessor import preprocess_for_display

DATA_DIR = Path("data/aptos/colored_images")
STAGE_NAMES = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}

print("Loading Models...")
pt_model = _load_pytorch_model()
tf_model = _load_tf_model()

print(f"{'='*60}\nEvaluating Models (Stage-by-Stage)\n{'='*60}")

pt_correct = Counter()
tf_correct = Counter()
total_images = Counter()

for stage in range(5):
    class_dir = DATA_DIR / str(stage)
    if not class_dir.exists():
        continue
        
    images = list(class_dir.glob("*.png"))[:50] # Test 50 images per stage
    
    for img_path in images:
        try:
            # We use preprocess_for_display to get both model inputs
            preprocessed_data = preprocess_for_display(str(img_path))
            
            # Pytorch model evaluation
            pt_result = _predict_pytorch(preprocessed_data["model_input_enhanced_highres"], pt_model) 
            
            # TF model evaluation
            tf_result = _predict_tensorflow(preprocessed_data["model_input"], tf_model)
            
            total_images[stage] += 1
            
            if pt_result["stage"] == stage:
                pt_correct[stage] += 1
                
            if tf_result["stage"] == stage:
                tf_correct[stage] += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print("\n[LATEST MODEL - EfficientNet-B3 PyTorch] Stage-by-Stage Accuracy:")
pt_overall_correct = 0
overall_total = 0
for stage in range(5):
    if total_images[stage] > 0:
        acc = (pt_correct[stage] / total_images[stage]) * 100
        print(f"  Stage {stage} ({STAGE_NAMES[stage]}): {acc:.1f}% ({pt_correct[stage]}/{total_images[stage]})")
        pt_overall_correct += pt_correct[stage]
        overall_total += total_images[stage]
print(f"  Overall Accuracy: {(pt_overall_correct/overall_total)*100:.1f}%")

print("\n[LAST MODEL - CNN TensorFlow] Stage-by-Stage Accuracy:")
tf_overall_correct = 0
for stage in range(5):
    if total_images[stage] > 0:
        acc = (tf_correct[stage] / total_images[stage]) * 100
        print(f"  Stage {stage} ({STAGE_NAMES[stage]}): {acc:.1f}% ({tf_correct[stage]}/{total_images[stage]})")
        tf_overall_correct += tf_correct[stage]
print(f"  Overall Accuracy: {(tf_overall_correct/overall_total)*100:.1f}%\n")
