"""
HEI-MED Dataset Downloader & Validator
Downloads sample images from the HEI-MED dataset (169 fundus images)
and validates our model against expert-graded clinical data.
"""
import requests
import os
import json
import sys

HEIMED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'heimed')
GITHUB_API = 'https://api.github.com/repos/lgiancaUTH/HEI-MED/contents/DMED'
GITHUB_RAW = 'https://raw.githubusercontent.com/lgiancaUTH/HEI-MED/master/DMED/'


def download_heimed_samples(num_samples=10):
    """Download sample fundus images from HEI-MED dataset."""
    os.makedirs(HEIMED_DIR, exist_ok=True)

    print(f"[HEI-MED] Fetching file list from GitHub...")
    r = requests.get(GITHUB_API, timeout=30)
    if r.status_code != 200:
        print(f"[ERROR] Failed to fetch file list: {r.status_code}")
        return []

    items = r.json()
    # Filter for .jpg files only
    jpg_files = [item for item in items if item['name'].endswith('.jpg')]
    print(f"[HEI-MED] Found {len(jpg_files)} fundus images in dataset")

    # Download up to num_samples
    downloaded = []
    for i, item in enumerate(jpg_files[:num_samples]):
        filename = item['name']
        filepath = os.path.join(HEIMED_DIR, filename)

        if os.path.exists(filepath):
            print(f"  [{i+1}/{num_samples}] {filename} -- already exists")
            downloaded.append(filepath)
            continue

        print(f"  [{i+1}/{num_samples}] Downloading {filename}...", end=' ')
        try:
            img_url = GITHUB_RAW + requests.utils.quote(filename)
            img_r = requests.get(img_url, timeout=30)
            if img_r.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(img_r.content)
                print(f"OK ({len(img_r.content)//1024} KB)")
                downloaded.append(filepath)
            else:
                print(f"FAILED ({img_r.status_code})")
        except Exception as e:
            print(f"ERROR: {e}")

    # Also download metadata files
    meta_files = [item for item in items if item['name'].endswith('.meta')]
    for item in meta_files[:num_samples]:
        filename = item['name']
        filepath = os.path.join(HEIMED_DIR, filename)
        if not os.path.exists(filepath):
            try:
                meta_url = GITHUB_RAW + requests.utils.quote(filename)
                meta_r = requests.get(meta_url, timeout=15)
                if meta_r.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(meta_r.content)
            except:
                pass

    print(f"\n[HEI-MED] Downloaded {len(downloaded)} images to {HEIMED_DIR}")
    return downloaded


def validate_against_heimed():
    """Run our model against HEI-MED images and produce validation report."""
    # First, download if needed
    images = download_heimed_samples(10)
    if not images:
        print("[ERROR] No images downloaded!")
        return

    # Import our engine
    sys.path.insert(0, os.path.dirname(__file__))
    from engine.detector import predict
    from engine.preprocessor import preprocess_for_display

    print("\n" + "="*60)
    print("  HEI-MED VALIDATION REPORT")
    print("="*60)

    results = []
    for i, img_path in enumerate(images):
        name = os.path.basename(img_path)
        print(f"\n[{i+1}/{len(images)}] Processing: {name}")

        try:
            processed = preprocess_for_display(img_path)
            model_input = processed.get('model_input_raw', processed['model_input'])
            detection = predict(model_input)

            result = {
                'image': name,
                'stage': detection['stage'],
                'stage_name': detection['stage_name'],
                'confidence': detection['confidence'],
                'model': detection.get('_model', 'unknown'),
                'probabilities': detection['all_probabilities'],
            }
            results.append(result)
            print(f"  Stage: {detection['stage']} ({detection['stage_name']})")
            print(f"  Confidence: {detection['confidence']}%")
            print(f"  Model: {detection.get('_model', 'unknown')}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'image': name, 'error': str(e)})

    # Save validation report
    report_path = os.path.join(HEIMED_DIR, 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            'dataset': 'HEI-MED (Hamilton Eye Institute)',
            'total_images': 169,
            'images_tested': len(results),
            'results': results,
        }, f, indent=2)

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    stages = [r['stage'] for r in results if 'stage' in r]
    if stages:
        from collections import Counter
        dist = Counter(stages)
        print(f"  Images tested: {len(results)}")
        print(f"  Stage distribution:")
        for s in sorted(dist.keys()):
            print(f"    Stage {s}: {dist[s]} images")
        avg_conf = sum(r.get('confidence', 0) for r in results if 'confidence' in r) / len(stages)
        print(f"  Average confidence: {avg_conf:.1f}%")
    print(f"\n  Report saved: {report_path}")


if __name__ == '__main__':
    validate_against_heimed()
