"""
Seed the database with HEI-MED sample data.
Creates patients with realistic names and runs analysis on downloaded images.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from database import create_patient, save_scan, init_db
from engine.preprocessor import preprocess_for_display
from engine.detector import predict
from engine.gradcam import generate_gradcam, get_heatmap_analysis
from engine.segmentor import segment_vessels
import cv2
import uuid

HEIMED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'heimed')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Realistic patient data (9 patients — Sunita Reddy removed)
PATIENTS = [
    {"name": "Rajesh Kumar", "age": 58, "gender": "Male", "diabetes_duration": 12, "sugar_level": 165, "hba1c": 8.2},
    {"name": "Anita Verma", "age": 45, "gender": "Female", "diabetes_duration": 6, "sugar_level": 130, "hba1c": 6.8},
    {"name": "Suresh Patel", "age": 62, "gender": "Male", "diabetes_duration": 18, "sugar_level": 190, "hba1c": 9.1},
    {"name": "Meera Joshi", "age": 52, "gender": "Female", "diabetes_duration": 8, "sugar_level": 145, "hba1c": 7.4},
    {"name": "Vikram Singh", "age": 67, "gender": "Male", "diabetes_duration": 22, "sugar_level": 210, "hba1c": 9.8},
    {"name": "Priya Sharma", "age": 41, "gender": "Female", "diabetes_duration": 4, "sugar_level": 120, "hba1c": 6.2},
    {"name": "Amit Desai", "age": 55, "gender": "Male", "diabetes_duration": 10, "sugar_level": 155, "hba1c": 7.9},
    {"name": "Kavita Rao", "age": 48, "gender": "Female", "diabetes_duration": 7, "sugar_level": 138, "hba1c": 7.0},
    {"name": "Deepak Gupta", "age": 60, "gender": "Male", "diabetes_duration": 15, "sugar_level": 178, "hba1c": 8.6},
]


def _build_report(detection, heatmap_analysis, vessel_stats, patient):
    """Build a realistic report without calling Gemma API."""
    stage = detection['stage']
    stage_name = detection['stage_name']
    conf = detection['confidence']

    diagnosis_map = {
        0: f"No Diabetic Retinopathy detected. Retinal scan appears normal with {conf}% confidence. No microaneurysms, hemorrhages, or exudates observed.",
        1: f"Mild Non-Proliferative Diabetic Retinopathy (NPDR) detected with {conf}% confidence. Few microaneurysms observed. Early-stage changes require monitoring.",
        2: f"Moderate NPDR detected with {conf}% confidence. Multiple microaneurysms and some retinal hemorrhages observed. Macular region shows early signs of stress.",
        3: f"Severe NPDR detected with {conf}% confidence. Extensive hemorrhages, venous beading, and intraretinal microvascular abnormalities (IRMA) observed.",
        4: f"Proliferative Diabetic Retinopathy (PDR) detected with {conf}% confidence. Neovascularization and potential vitreous hemorrhage identified. Urgent referral needed.",
    }

    coverage = heatmap_analysis.get('activation_coverage', 0)
    density = vessel_stats.get('vessel_density_percent', 0)

    heatmap_text = (
        f"AI attention heatmap shows {coverage}% activation coverage. "
        f"{'Minimal focal areas of concern.' if coverage < 20 else 'Moderate activation in the macular and peripapillary regions.' if coverage < 40 else 'Significant activation across retinal surface indicating widespread pathology.'}"
    )

    vessel_text = (
        f"Vessel density measured at {density}%. "
        f"{'Normal vascular architecture observed.' if density > 8 else 'Reduced vessel density may indicate ischemic changes.' if density > 4 else 'Critically low vessel density — possible vascular occlusion.'}"
    )

    risk_6m = {0: "3-5%", 1: "8-12%", 2: "15-20%", 3: "25-35%", 4: "40-50%"}
    risk_12m = {0: "5-10%", 1: "12-18%", 2: "22-30%", 3: "35-50%", 4: "50-70%"}

    actions_map = {
        0: ["Continue annual retinal screening", "Maintain blood sugar below 140 mg/dL", "Monitor HbA1c every 3 months", "30 mins daily walking or exercise"],
        1: ["Schedule follow-up in 6 months", "Target HbA1c below 7.0%", "Blood pressure monitoring", "Start diabetic-friendly diet"],
        2: ["Urgent referral to ophthalmologist within 1 month", "Strict blood sugar control required", "Consider laser photocoagulation evaluation", "Daily blood glucose monitoring"],
        3: ["Immediate ophthalmologist referral within 2 weeks", "Anti-VEGF injection evaluation", "Strict glycemic control mandatory", "Weekly self-monitoring of vision"],
        4: ["Emergency ophthalmology consultation within 48 hours", "Pan-retinal photocoagulation likely needed", "Vitrectomy evaluation", "Hospital-based treatment plan"],
    }

    diet_map = {
        0: ["Green leafy vegetables (spinach, methi)", "Omega-3 rich foods (walnuts, flaxseed)", "Low glycemic index foods", "Limit sugar and refined carbs"],
        1: ["Increase fiber intake (oats, dal, brown rice)", "Antioxidant-rich berries and fruits", "Reduce sodium to below 2300mg/day", "Avoid processed and fried foods"],
        2: ["Strict diabetic diet plan required", "Include turmeric and cinnamon (anti-inflammatory)", "Limit carbohydrate portions", "Protein-rich meals (paneer, egg whites, lentils)"],
        3: ["Consult dietitian for personalized meal plan", "Vitamin A and C rich foods for retinal health", "Zero refined sugar policy", "Small frequent meals every 3 hours"],
        4: ["Hospital dietitian supervision recommended", "High antioxidant supplementation", "Controlled protein intake", "Hydration therapy — 3L water daily"],
    }

    followup_map = {
        0: "Annual screening recommended. Next visit: 12 months.",
        1: "Follow-up in 6 months. Monitor vision changes weekly.",
        2: "Follow-up in 2-3 months. Ophthalmologist referral needed.",
        3: "Follow-up in 2-4 weeks. Urgent specialist evaluation.",
        4: "Immediate follow-up within 48 hours. Emergency referral.",
    }

    return {
        "diagnosis": diagnosis_map.get(stage, "Analysis complete."),
        "heatmap_analysis": heatmap_text,
        "vessel_analysis": vessel_text,
        "risk_prediction": {
            "6_month": {
                "progression_probability": risk_6m.get(stage, "—"),
                "if_untreated": f"Risk of progression to Stage {min(stage+1, 4)} if blood sugar remains uncontrolled.",
                "if_managed": "Stable or improved with proper glycemic control and lifestyle changes."
            },
            "12_month": {
                "progression_probability": risk_12m.get(stage, "—"),
                "if_untreated": f"Significant risk of advancing to Stage {min(stage+1, 4)} without intervention.",
                "if_managed": "High probability of stabilization with consistent treatment adherence."
            }
        },
        "action_plan": actions_map.get(stage, []),
        "diet_recommendations": diet_map.get(stage, []),
        "follow_up": followup_map.get(stage, "Consult your doctor."),
    }


def seed_database():
    """Create patients and run scans on HEI-MED images."""
    init_db()

    images = sorted([f for f in os.listdir(HEIMED_DIR) if f.endswith('.jpg')])
    if not images:
        print("[ERROR] No HEI-MED images found in", HEIMED_DIR)
        return

    # Only use as many images as patients
    images = images[:len(PATIENTS)]

    print("=" * 60)
    print("  SEEDING DATABASE WITH HEI-MED DATA")
    print(f"  {len(PATIENTS)} patients, {len(images)} images")
    print("=" * 60)

    for i, (img_name, patient_data) in enumerate(zip(images, PATIENTS)):
        img_path = os.path.join(HEIMED_DIR, img_name)
        print(f"\n[{i+1}/{len(PATIENTS)}] {patient_data['name']} -- {img_name}")

        try:
            patient = create_patient(**patient_data)
            pid = patient['id']
            print(f"  Created patient: {pid}")

            processed = preprocess_for_display(img_path)
            model_input = processed["model_input"]
            model_input_raw = processed["model_input_raw"]
            original = processed["original"]

            analysis_id = str(uuid.uuid4())[:12]
            detection = predict(model_input_raw)
            print(f"  Detection: Stage {detection['stage']} ({detection['stage_name']}) -- {detection['confidence']}%")

            original_path = os.path.join(RESULTS_DIR, f"{analysis_id}_scan.png")
            cv2.imwrite(original_path, original)

            heatmap_path = os.path.join(RESULTS_DIR, f"{analysis_id}_heatmap.png")
            _, heatmap_raw = generate_gradcam(model_input, original, save_path=heatmap_path)
            heatmap_analysis = get_heatmap_analysis(heatmap_raw)

            vessel_path = os.path.join(RESULTS_DIR, f"{analysis_id}_vessels.png")
            _, vessel_stats = segment_vessels(original, save_path=vessel_path)

            # Build full report (without Gemma API)
            report = _build_report(detection, heatmap_analysis, vessel_stats, patient_data)

            image_paths = {
                "original": f"/results/{analysis_id}_scan.png",
                "heatmap": f"/results/{analysis_id}_heatmap.png",
                "vessels": f"/results/{analysis_id}_vessels.png",
            }

            save_scan(
                scan_id=analysis_id,
                patient_id=pid,
                detection_result=detection,
                heatmap_analysis=heatmap_analysis,
                vessel_stats=vessel_stats,
                report=report,
                image_paths=image_paths,
                processing_time=round(2.5 + i * 0.3, 1)
            )
            print("  [OK] Saved to DB!")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"  DONE! Database seeded with {len(PATIENTS)} patients.")
    print("=" * 60)


if __name__ == '__main__':
    seed_database()
