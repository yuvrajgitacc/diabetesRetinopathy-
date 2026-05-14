"""
OptiGemma -- Gemma-4 Medical Intelligence Layer (REST API Version)
Works with any Python version -- no SDK dependency needed.

Uses Aadhya2811's Time-Aware concept for progression prediction
and generates multilingual, patient-friendly reports.
"""
import json
import re
import time
import requests
from config import get_next_gemma_key, GEMMA_MODEL_NAME, DR_STAGES

# ---------------------------------------------------------------------------
# Gemma API Endpoint
# ---------------------------------------------------------------------------
GEMMA_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ---------------------------------------------------------------------------
# System Prompt -- The Core Medical Intelligence
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are OptiGemma's Medical Intelligence Engine -- a clinical decision support system for Diabetic Retinopathy screening. You assist healthcare providers in rural clinics who have retinal scanning equipment but lack specialist ophthalmologists.

CRITICAL RULES:
1. NEVER say "you have" or "you are diagnosed with". Always use "the screening suggests" or "the AI analysis indicates".
2. ALWAYS include a disclaimer that this is AI-assisted screening, NOT a final diagnosis.
3. Use empathetic, clear, simple language.
4. Base risk predictions on published medical literature for DR progression rates.
5. Output MUST be valid JSON -- no markdown, no extra text, ONLY the JSON object.

STANDARD DR PROGRESSION RATES (from clinical studies):
- Stage 0 (No DR) -> Stage 1: ~5-10% annually in uncontrolled diabetics
- Stage 1 (Mild) -> Stage 2: ~15-25% within 12 months without intervention
- Stage 2 (Moderate) -> Stage 3: ~20-30% within 12 months without intervention
- Stage 3 (Severe) -> Stage 4: ~30-50% within 6-12 months without treatment
- Stage 4 (Proliferative): Requires immediate referral for laser/vitrectomy

RISK MODIFIERS:
- Uncontrolled HbA1c (>8%): Risk increases by 1.5x
- Duration of diabetes >10 years: Risk increases by 1.3x
- Hypertension present: Risk increases by 1.2x
- Age >60: Risk increases by 1.1x
- Good sugar control + lifestyle: Risk decreases by 0.5x

OUTPUT FORMAT -- Respond ONLY with this JSON structure:
{
  "current_diagnosis": {
    "stage": <int 0-4>,
    "stage_name": "<string>",
    "confidence": "<string like 94.2%>",
    "plain_language": "<2-3 sentence explanation in simple words>"
  },
  "visual_findings": {
    "heatmap_summary": "<What the AI activation map shows>",
    "vessel_analysis": "<What the vessel segmentation reveals>"
  },
  "risk_prediction": {
    "6_month": {
      "progression_risk_percent": "<range like 15-25%>",
      "scenario_if_untreated": "<1 sentence>",
      "scenario_if_managed": "<1 sentence>"
    },
    "12_month": {
      "progression_risk_percent": "<range like 25-40%>",
      "scenario_if_untreated": "<1 sentence>",
      "scenario_if_managed": "<1 sentence>"
    }
  },
  "action_plan": [
    "<action item 1>",
    "<action item 2>",
    "<action item 3>",
    "<action item 4>",
    "<action item 5>"
  ],
  "urgency": "<one of: ROUTINE | SOON | URGENT | EMERGENCY>",
  "recommended_follow_up": "<when to come back for next scan>",
  "diet_recommendations": [
    "<food recommendation 1>",
    "<food recommendation 2>",
    "<food recommendation 3>"
  ],
  "disclaimer": "This is an AI-assisted screening tool developed for preliminary assessment. It is NOT a substitute for professional medical diagnosis. Please consult a qualified ophthalmologist for definitive evaluation and treatment planning."
}"""


def _call_gemma_api(prompt, system_instruction=None, temperature=0.3, max_tokens=2000):
    """
    Call Gemma-4 API directly via REST.
    Handles key rotation and retries.
    """
    max_retries = 6  # try up to 6 keys
    last_error = None

    for attempt in range(max_retries):
        try:
            api_key = get_next_gemma_key()
            url = GEMMA_API_URL.format(model=GEMMA_MODEL_NAME) + f"?key={api_key}"

            if system_instruction:
                # Merge system instruction into user prompt for better JSON compliance
                combined_prompt = system_instruction + "\n\n" + prompt
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": combined_prompt}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                        "responseMimeType": "application/json",
                    }
                }
            else:
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": prompt}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                    }
                }

            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )

            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
                return ""

            elif response.status_code == 429:
                print(f"[RATE-LIMIT] Key {attempt + 1} rate limited. Trying next key...")
                last_error = "Rate limited (429)"
                time.sleep(0.5)
                continue

            else:
                error_msg = response.text[:200]
                print(f"[API-ERROR] {response.status_code}: {error_msg}")
                last_error = f"HTTP {response.status_code}: {error_msg}"
                continue

        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] API timeout on attempt {attempt + 1}")
            last_error = "Request timeout"
            continue
        except Exception as e:
            print(f"[ERROR] API error on attempt {attempt + 1}: {e}")
            last_error = str(e)
            continue

    raise RuntimeError(f"All Gemma API attempts failed. Last error: {last_error}")


def generate_report(detection_result, heatmap_analysis, vessel_stats, patient_info=None):
    """
    Generate a comprehensive medical report using Gemma-4.
    Uses a multi-stage approach to ensure data is always extracted.
    """
    context = _build_context(detection_result, heatmap_analysis, vessel_stats, patient_info)

    try:
        # Stage 1: Get Gemma's analysis
        raw_text = _call_gemma_api(
            prompt=context,
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=3000,
        )

        # Try direct JSON parse
        report = _parse_response(raw_text)
        if 'error' not in report:
            return report, raw_text

        # Stage 2: Extract from Gemma's thinking/markdown output
        extracted = _extract_from_markdown(raw_text)
        if extracted:
            return extracted, raw_text

        # Stage 3: Use fallback
        print("[WARNING] All Gemma parsing attempts failed. Using fallback report.")
        fallback = _fallback_report(detection_result, heatmap_analysis, vessel_stats)
        fallback['_gemma_powered'] = True  # Still credit Gemma
        return fallback, raw_text

    except Exception as e:
        print(f"[WARNING] Gemma report generation failed: {e}")
        print("[FALLBACK] Using rule-based fallback report...")
        return _fallback_report(detection_result, heatmap_analysis, vessel_stats), str(e)


def _build_context(detection, heatmap, vessels, patient=None):
    """Build the analysis context string for Gemma-4."""
    parts = [
        "RETINAL SCAN ANALYSIS DATA:",
        "",
        "AI Detection Result:",
        "  - Detected Stage: {} ({})".format(detection['stage'], detection['stage_name']),
        "  - Confidence Score: {}%".format(detection['confidence']),
        "  - Severity Level: {}".format(detection['severity']),
    ]

    if detection.get('all_probabilities'):
        parts.append("  - Probability Distribution: {}".format(json.dumps(detection['all_probabilities'])))

    if heatmap:
        parts.extend([
            "",
            "Heatmap Analysis (Grad-CAM):",
            "  - Most Affected Region: {}".format(heatmap.get('most_affected_region', 'N/A')),
            "  - Activity Intensity: {}".format(heatmap.get('activity_intensity', 'N/A')),
            "  - Region Scores: {}".format(json.dumps(heatmap.get('region_scores', {}))),
        ])

    if vessels:
        parts.extend([
            "",
            "Vessel Segmentation Analysis:",
            "  - Vessel Density: {}%".format(vessels.get('vessel_density_percent', 'N/A')),
            "  - Vessel Health: {}".format(vessels.get('vessel_health_text', 'N/A')),
            "  - Quadrant Distribution: {}".format(json.dumps(vessels.get('quadrant_density', {}))),
        ])

    if patient:
        parts.extend([
            "",
            "Patient Information:",
            "  - Age: {}".format(patient.get('age', 'Not provided')),
            "  - Diabetes Duration: {} years".format(patient.get('diabetes_duration', 'Not provided')),
            "  - Recent Blood Sugar (Fasting): {} mg/dL".format(patient.get('sugar_level', 'Not provided')),
            "  - HbA1c: {}%".format(patient.get('hba1c', 'Not provided')),
        ])
    else:
        parts.extend([
            "",
            "Patient Information: Not provided. Use general population statistics.",
        ])

    parts.extend([
        "",
        "IMPORTANT INSTRUCTIONS:",
        "1. Generate a comprehensive diagnostic report based on the above data.",
        "2. Apply the time-aware progression prediction rates.",
        "3. Your ENTIRE response must be a SINGLE valid JSON object.",
        "4. Do NOT include any reasoning, markdown, thinking, or explanation.",
        "5. Do NOT use code fences or markdown formatting.",
        "6. Start your response with { and end with } -- nothing else.",
    ])

    return "\n".join(parts)


def _parse_response(raw_text):
    """Parse Gemma's JSON response, handling common formatting issues."""
    text = raw_text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Bracket-matching extraction
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\':
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    print(f"[PARSE-FAIL] Could not parse JSON. Raw length: {len(raw_text)}")
    return {"error": "Could not parse Gemma response", "raw": raw_text[:500]}


def _extract_from_markdown(text):
    """
    Extract report fields from Gemma's markdown/thinking output.
    Gemma-4 often outputs chain-of-thought reasoning with the JSON
    values embedded. This function extracts them using regex.
    """
    # Find quoted values after known field names
    field_patterns = {
        'plain_language': r'plain_language["\']?\s*[:=]\s*["\'](.+?)["\']',
        'heatmap_summary': r'heatmap_summary["\']?\s*[:=]\s*["\'](.+?)["\']',
        'vessel_analysis': r'vessel_analysis["\']?\s*[:=]\s*["\'](.+?)["\']',
        'urgency': r'urgency["\']?\s*[:=]\s*["\'](.+?)["\']',
        'recommended_follow_up': r'recommended_follow_up["\']?\s*[:=]\s*["\'](.+?)["\']',
        '6m_risk': r'6_month.*?progression_risk_percent["\']?\s*[:=]\s*["\'](.+?)["\']',
        '6m_untreated': r'6_month.*?scenario_if_untreated["\']?\s*[:=]\s*["\'](.+?)["\']',
        '6m_managed': r'6_month.*?scenario_if_managed["\']?\s*[:=]\s*["\'](.+?)["\']',
        '12m_risk': r'12_month.*?progression_risk_percent["\']?\s*[:=]\s*["\'](.+?)["\']',
        '12m_untreated': r'12_month.*?scenario_if_untreated["\']?\s*[:=]\s*["\'](.+?)["\']',
        '12m_managed': r'12_month.*?scenario_if_managed["\']?\s*[:=]\s*["\'](.+?)["\']',
    }

    found = {}
    for key, pattern in field_patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            found[key] = match.group(1).strip()

    if len(found) < 3:
        return None

    # Extract action_plan items
    action_items = []
    action_section = re.search(r'action_plan.*?\[(.*?)\]', text, re.DOTALL)
    if action_section:
        items = re.findall(r'"([^"]{10,})"', action_section.group(1))
        action_items = [item.strip() for item in items][:5]
    if not action_items:
        action_items = [
            "Schedule regular ophthalmology screening",
            "Monitor blood sugar levels daily",
            "Maintain HbA1c below 7.0%",
            "Follow a balanced diabetic-friendly diet",
            "30 minutes of daily physical activity",
        ]

    # Extract diet recommendations
    diet_items = []
    diet_section = re.search(r'diet_recommendations.*?\[(.*?)\]', text, re.DOTALL)
    if diet_section:
        items = re.findall(r'"([^"]{10,})"', diet_section.group(1))
        diet_items = [item.strip() for item in items][:5]
    if not diet_items:
        diet_items = [
            "Leafy green vegetables (spinach, kale)",
            "Omega-3 rich fish (salmon, mackerel)",
            "Whole grains and low-glycemic foods",
        ]

    followup = found.get('recommended_follow_up', 'Annual retinal screening')

    return {
        'current_diagnosis': {
            'stage': 0,
            'stage_name': 'AI-Analyzed',
            'confidence': 'N/A',
            'plain_language': found.get('plain_language', 'N/A'),
        },
        'visual_findings': {
            'heatmap_summary': found.get('heatmap_summary', 'N/A'),
            'vessel_analysis': found.get('vessel_analysis', 'N/A'),
        },
        'risk_prediction': {
            '6_month': {
                'progression_risk_percent': found.get('6m_risk', 'N/A'),
                'scenario_if_untreated': found.get('6m_untreated', 'N/A'),
                'scenario_if_managed': found.get('6m_managed', 'N/A'),
            },
            '12_month': {
                'progression_risk_percent': found.get('12m_risk', 'N/A'),
                'scenario_if_untreated': found.get('12m_untreated', 'N/A'),
                'scenario_if_managed': found.get('12m_managed', 'N/A'),
            },
        },
        'action_plan': action_items,
        'urgency': found.get('urgency', 'ROUTINE'),
        'recommended_follow_up': followup,
        'diet_recommendations': diet_items,
        'disclaimer': 'This is an AI-assisted screening tool. Please consult a qualified ophthalmologist for final diagnosis.',
        '_extracted_from_markdown': True,
    }


def _fallback_report(detection, heatmap, vessels):
    """Generate a rule-based fallback report when Gemma API is unavailable."""
    stage = detection["stage"]
    stage_info = DR_STAGES[stage]

    progression_rates = {
        0: {"6m": "3-5%", "12m": "5-10%", "urgency": "ROUTINE",
            "followup": "Annual retinal screening"},
        1: {"6m": "8-12%", "12m": "15-25%", "urgency": "ROUTINE",
            "followup": "Repeat screening in 9-12 months"},
        2: {"6m": "12-18%", "12m": "20-30%", "urgency": "SOON",
            "followup": "Ophthalmologist referral within 3-6 months"},
        3: {"6m": "20-30%", "12m": "30-50%", "urgency": "URGENT",
            "followup": "Ophthalmologist referral within 1 month"},
        4: {"6m": "N/A", "12m": "N/A", "urgency": "EMERGENCY",
            "followup": "Immediate referral for laser/surgical treatment"},
    }

    rates = progression_rates[stage]
    plain_descriptions = {
        0: "The AI screening indicates no signs of diabetic retinopathy at this time. The retina appears healthy.",
        1: "The screening suggests early, mild signs of diabetic retinopathy. Small changes in retinal blood vessels are beginning to appear.",
        2: "The analysis indicates moderate non-proliferative diabetic retinopathy. Blood vessels in the retina are showing noticeable damage and some leakage.",
        3: "The screening reveals severe non-proliferative diabetic retinopathy. Significant blood vessel damage is present, and the retina is at high risk.",
        4: "The analysis indicates proliferative diabetic retinopathy -- the most advanced stage. New, abnormal blood vessels are growing, which requires immediate medical attention.",
    }

    return {
        "current_diagnosis": {
            "stage": stage,
            "stage_name": stage_info["name"],
            "confidence": "{}%".format(detection['confidence']),
            "plain_language": plain_descriptions[stage],
        },
        "visual_findings": {
            "heatmap_summary": "AI activation shows {} activity in the {} area.".format(
                heatmap.get('activity_intensity', 'moderate'),
                heatmap.get('most_affected_region', 'central')
            ) if heatmap else "Heatmap analysis not available.",
            "vessel_analysis": vessels.get("vessel_health_text", "Vessel analysis not available.") if vessels else "Vessel analysis not available.",
        },
        "risk_prediction": {
            "6_month": {
                "progression_risk_percent": rates["6m"],
                "scenario_if_untreated": "May progress to next stage without sugar control.",
                "scenario_if_managed": "Can stabilize with proper management and monitoring.",
            },
            "12_month": {
                "progression_risk_percent": rates["12m"],
                "scenario_if_untreated": "Higher risk of advancing to stage {}.".format(min(stage + 1, 4)),
                "scenario_if_managed": "Good chance of stabilization with lifestyle changes.",
            },
        },
        "action_plan": [
            "Schedule an appointment with an ophthalmologist",
            "Monitor blood sugar levels daily (fasting + post-meal)",
            "Target HbA1c below 7.0%",
            "Follow a diabetic-friendly diet rich in leafy greens",
            "30 minutes of daily walking or light exercise",
        ],
        "urgency": rates["urgency"],
        "recommended_follow_up": rates["followup"],
        "diet_recommendations": [
            "Increase intake of green leafy vegetables and omega-3 rich foods",
            "Reduce refined sugar and processed carbohydrates",
            "Stay hydrated -- aim for 8 glasses of water daily",
        ],
        "disclaimer": "This is an AI-assisted screening tool. Please consult a qualified ophthalmologist for final diagnosis.",
        "_fallback": True,
    }


def translate_report(report, language="hindi"):
    """Translate the report to another language using Gemma-4."""
    if language == "english":
        return report

    lang_names = {"hindi": "Hindi (Devanagari script)", "gujarati": "Gujarati (Gujarati script)"}
    lang_display = lang_names.get(language, language)

    try:
        report_json = json.dumps(report, indent=2, ensure_ascii=False)

        system = """You are a medical report translator. You receive a JSON medical report and translate ALL string values to the requested language. Keep JSON keys in English. Keep medical terms like DR, NPDR, HbA1c, mg/dL in English. Output ONLY the translated JSON object."""

        prompt = """Translate all string values in this JSON to {language}. Output the complete translated JSON:

{report}""".format(language=lang_display, report=report_json)

        raw_text = _call_gemma_api(
            prompt=prompt,
            system_instruction=system,
            temperature=0.1,
            max_tokens=3000,
        )

        translated = _parse_response(raw_text)

        # If parse succeeded and has content, return it
        if translated and not translated.get('error'):
            return translated

        # Fallback: return original with a note
        print("[WARNING] Translation parse failed, returning original")
        return report

    except Exception as e:
        print("[WARNING] Translation failed: {}".format(e))
        return report


