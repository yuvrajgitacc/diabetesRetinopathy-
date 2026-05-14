"""Debug: Check what Gemma actually returns."""
import requests
import json

url = "http://127.0.0.1:5000/analyze"
with open("sample_data/test_fundus.jpg", "rb") as f:
    files = {"image": ("test.jpg", f, "image/jpeg")}
    data = {"age": "55", "diabetes_duration": "10", "sugar_level": "180", "hba1c": "8.2"}
    r = requests.post(url, files=files, data=data, timeout=120)

result = r.json()
print("=== FULL REPORT JSON ===")
print(json.dumps(result.get("report", {}), indent=2))
