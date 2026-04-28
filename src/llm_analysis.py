from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import requests

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def build_prompt(prediction_result: dict[str, Any]) -> str:
    return f"""
You are a network security analyst assistant.
Review the following intrusion-detection result and return a JSON object.

Model: {prediction_result['model_name']}
Prediction text: {prediction_result['prediction_text']}
Prediction label: {prediction_result['prediction_label']}
Prediction score: {prediction_result.get('prediction_score')}
Key features: {json.dumps(prediction_result.get('key_features', {}), ensure_ascii=False)}
Raw feature summary: {json.dumps(prediction_result.get('raw_features', {}), ensure_ascii=False)}

Required JSON fields:
- risk_level
- explanation
- impact
- suggestion
""".strip()


def call_llm_api(prompt: str) -> dict[str, Any]:
    if not config.LLM_API_KEY:
        return {}

    response = requests.post(
        f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a network security assistant. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def parse_llm_response(response: dict[str, Any]) -> dict[str, str]:
    if not response:
        return {}

    content = response["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "risk_level": "Unknown",
            "explanation": content,
            "impact": "Manual review is still required.",
            "suggestion": "Cross-check the raw traffic and system logs before taking action.",
        }


def fallback_analysis(prediction_result: dict[str, Any]) -> dict[str, str]:
    score = float(prediction_result.get("prediction_score") or 0)
    prediction_text = prediction_result.get("prediction_text", "Unknown")
    key_features = prediction_result.get("key_features", {})

    if prediction_text == "Attack":
        if score >= 0.85:
            risk_level = "High"
        elif score >= 0.65:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        explanation = (
            "The model classified this traffic as an attack because several key "
            f"features appear abnormal, including {', '.join(key_features.keys()) if key_features else 'multiple network behavior indicators'}."
        )
        impact = "This traffic may lead to unauthorized access, abnormal resource consumption, or service disruption."
        suggestion = "Inspect the source IP, session behavior, and target host state, then apply firewall or intrusion-prevention controls as needed."
    else:
        risk_level = "Low"
        explanation = "The model classified this traffic as normal and did not detect strong attack indicators."
        impact = "The short-term risk appears low, but continued monitoring is still recommended."
        suggestion = "Keep the relevant logs and continue observing similar sessions for abnormal changes."

    return {
        "risk_level": risk_level,
        "explanation": explanation,
        "impact": impact,
        "suggestion": suggestion,
    }


def analyze_prediction_result(prediction_result: dict[str, Any]) -> dict[str, str]:
    prompt = build_prompt(prediction_result)
    try:
        response = call_llm_api(prompt)
        parsed = parse_llm_response(response)
        if parsed:
            return parsed
    except Exception:
        pass
    return fallback_analysis(prediction_result)
