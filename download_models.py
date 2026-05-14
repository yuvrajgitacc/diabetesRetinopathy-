"""Download model files from GitHub repos."""
import requests
import os
import sys

def download_file(url, dest_path, name):
    """Download a file with progress."""
    print(f"[DOWNLOAD] {name}...")
    print(f"  URL: {url}")
    print(f"  Dest: {dest_path}")
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=120, allow_redirects=True)
        response.raise_for_status()
        
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded / total * 100)
                    if pct % 20 == 0:
                        print(f"  {pct}% ({downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB)")
        
        size_mb = os.path.getsize(dest_path) / 1024 / 1024
        print(f"  [OK] Downloaded {size_mb:.1f} MB")
        return True
        
    except Exception as e:
        print(f"  [FAILED] {e}")
        return False


# Tanwar-12 model.h5
download_file(
    "https://github.com/Tanwar-12/DIABETIC-RETIONOPATHY-DETECTION/raw/main/Deploy/diabetic-retinopathy-detection/Model/model.h5",
    "models/model.h5",
    "Tanwar-12 ResNet50 DR Classification Model"
)

# RishiSwethan vessel segmentation model
download_file(
    "https://github.com/rishiswethan/Diabetic-Retinopathy-Detection-Retinal-Vessel-Segmentation/raw/main/trained_models/segment_best_val_loss.pth",
    "models/vessel_model/segment_best_val_loss.pth",
    "RishiSwethan Vessel Segmentation Model"
)

# RishiSwethan classification model (backup)
download_file(
    "https://github.com/rishiswethan/Diabetic-Retinopathy-Detection-Retinal-Vessel-Segmentation/raw/main/trained_models/best_val_loss.pt",
    "models/vessel_model/best_val_loss.pt",
    "RishiSwethan Classification Model"
)

# RishiSwethan hyperparams
download_file(
    "https://github.com/rishiswethan/Diabetic-Retinopathy-Detection-Retinal-Vessel-Segmentation/raw/main/trained_models/best_hp.json",
    "models/vessel_model/best_hp.json",
    "RishiSwethan Hyperparameters"
)

print("\n[DONE] All downloads attempted. Check above for any failures.")
