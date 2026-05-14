"""
Download Diabetic Retinopathy training dataset.

Tries multiple sources:
1. Kaggle CLI (if authenticated)
2. Hugging Face Hub (free, no auth needed for many DR datasets)
3. Manual instructions
"""
import os
import sys
import subprocess
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "aptos")
os.makedirs(DATA_DIR, exist_ok=True)


def try_kaggle():
    """Try downloading via Kaggle CLI."""
    print("[1/3] Trying Kaggle CLI...")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "download",
             "-c", "aptos2019-blindness-detection", "-p", DATA_DIR],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("[OK] Downloaded from Kaggle!")
            # Extract
            for f in os.listdir(DATA_DIR):
                if f.endswith('.zip'):
                    zpath = os.path.join(DATA_DIR, f)
                    print(f"  Extracting {f}...")
                    with zipfile.ZipFile(zpath, 'r') as z:
                        z.extractall(DATA_DIR)
                    os.remove(zpath)
            return True
        else:
            print(f"  Kaggle failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  Kaggle error: {e}")
        return False


def try_huggingface():
    """Download from Hugging Face datasets."""
    print("[2/3] Trying Hugging Face Hub...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
                       capture_output=True, timeout=60)

        from huggingface_hub import hf_hub_download, list_repo_files

        REPO = "sartajbhuvaji/Brain-Tumor-Classification"
        # Try a known DR dataset on HF
        DR_REPOS = [
            "aharley/aptos2019",
            "marmal88/skin_cancer",
        ]

        # Use datasets library for the standard APTOS dataset
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"],
                       capture_output=True, timeout=120)

        from datasets import load_dataset
        print("  Loading APTOS-like DR dataset from HuggingFace...")

        # Try loading a DR dataset
        try:
            ds = load_dataset("alkzar90/diabetic-retinopathy-dataset", split="train")
            print(f"  Found {len(ds)} images!")

            # Save to folder structure
            for i, sample in enumerate(ds):
                label = sample["diagnosis"]
                label_dir = os.path.join(DATA_DIR, str(label))
                os.makedirs(label_dir, exist_ok=True)
                img = sample["image"]
                img.save(os.path.join(label_dir, f"img_{i:05d}.png"))
                if (i + 1) % 500 == 0:
                    print(f"  Saved {i+1}/{len(ds)} images...")

            print(f"[OK] Downloaded {len(ds)} images to {DATA_DIR}")
            return True
        except Exception as e2:
            print(f"  HF dataset error: {e2}")

        return False

    except Exception as e:
        print(f"  HuggingFace error: {e}")
        return False


def try_direct_download():
    """Try direct download from public mirrors."""
    print("[3/3] Trying direct download...")
    import requests

    # Try GitHub-hosted preprocessed datasets
    urls = [
        # Preprocessed APTOS (resized) — commonly shared
        "https://github.com/JordiCorbilla/ocular-disease-intelligent-recognition-deep-learning/raw/master/odir_training_data.zip",
    ]

    for url in urls:
        try:
            print(f"  Trying: {url[:60]}...")
            r = requests.head(url, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                size_mb = int(r.headers.get('content-length', 0)) / 1e6
                print(f"  Found! Size: {size_mb:.0f} MB. Downloading...")
                r = requests.get(url, stream=True, timeout=600)
                zpath = os.path.join(DATA_DIR, "dataset.zip")
                with open(zpath, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                with zipfile.ZipFile(zpath) as z:
                    z.extractall(DATA_DIR)
                os.remove(zpath)
                print("[OK] Downloaded and extracted!")
                return True
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    return False


def print_manual_instructions():
    """Print manual download instructions."""
    print("\n" + "=" * 60)
    print("  MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print()
    print("Automatic download failed. Please download manually:")
    print()
    print("OPTION A: Kaggle (recommended, 3662 images)")
    print("  1. Run: kaggle auth login")
    print("  2. Follow the browser OAuth flow")
    print("  3. Then re-run this script")
    print()
    print("OPTION B: Manual Kaggle download")
    print("  1. Visit: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data")
    print("  2. Download 'train.csv' and 'train_images.zip'")
    print(f"  3. Extract to: {DATA_DIR}")
    print("  4. Ensure structure:")
    print(f"       {DATA_DIR}/train.csv")
    print(f"       {DATA_DIR}/train_images/*.png")
    print()
    print("OPTION C: Folder structure (any DR images)")
    print(f"  Place images in: {DATA_DIR}/0/ {DATA_DIR}/1/ ... {DATA_DIR}/4/")
    print("  (folder name = DR stage 0-4)")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("  OptiGemma — Dataset Downloader")
    print("=" * 60)

    if try_kaggle():
        print("\n✅ Dataset ready! Run: python train_model.py")
        sys.exit(0)

    if try_huggingface():
        print("\n✅ Dataset ready! Run: python train_model.py")
        sys.exit(0)

    if try_direct_download():
        print("\n✅ Dataset ready! Run: python train_model.py")
        sys.exit(0)

    print_manual_instructions()
    sys.exit(1)
