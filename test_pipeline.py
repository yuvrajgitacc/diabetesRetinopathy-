"""Quick test of the full OptiGemma pipeline."""
import requests
import json
import os

# Use the test image
image_path = "sample_data/test_fundus.jpg"
if not os.path.exists(image_path):
    print("[ERROR] Test image not found!")
    exit(1)

url = "http://127.0.0.1:5000/analyze"

with open(image_path, "rb") as f:
    files = {"image": ("test_fundus.jpg", f, "image/jpeg")}
    data = {
        "age": "55",
        "diabetes_duration": "10",
        "sugar_level": "180",
        "hba1c": "8.2",
    }
    
    print("[TEST] Sending analysis request...")
    response = requests.post(url, files=files, data=data, timeout=120)

print(f"[STATUS] {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"\n[DETECTION]")
    print(f"  Stage: {result['detection']['stage']} - {result['detection']['stage_name']}")
    print(f"  Confidence: {result['detection']['confidence']}%")
    print(f"  Mock: {result['detection'].get('_mock', False)}")
    
    print(f"\n[HEATMAP]")
    print(f"  Most affected: {result['heatmap_analysis']['most_affected_region']}")
    print(f"  Intensity: {result['heatmap_analysis']['activity_intensity']}")
    
    print(f"\n[VESSELS]")
    print(f"  Density: {result['vessel_stats']['vessel_density_percent']}%")
    print(f"  Health: {result['vessel_stats']['vessel_health_text']}")
    
    print(f"\n[REPORT]")
    report = result['report']
    print(f"  Fallback: {report.get('_fallback', False)}")
    if 'current_diagnosis' in report:
        print(f"  Diagnosis: {report['current_diagnosis'].get('plain_language', 'N/A')[:100]}")
    if 'urgency' in report:
        print(f"  Urgency: {report['urgency']}")
    if 'risk_prediction' in report:
        r6 = report['risk_prediction'].get('6_month', {})
        print(f"  6-month risk: {r6.get('progression_risk_percent', 'N/A')}")
    
    print(f"\n[IMAGES]")
    for k, v in result['images'].items():
        print(f"  {k}: {v}")
    
    print(f"\n[TIME] {result['processing_time']}s")
    print("\n[SUCCESS] Full pipeline working!")
else:
    print(f"[ERROR] {response.text[:500]}")
