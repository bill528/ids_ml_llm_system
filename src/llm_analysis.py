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
你是一名网络安全分析助手。请根据以下网络入侵检测结果，输出结构化安全分析。

模型名称：{prediction_result['model_name']}
检测结论：{prediction_result['prediction_text']}
预测标签：{prediction_result['prediction_label']}
预测分数：{prediction_result.get('prediction_score')}
关键特征：{json.dumps(prediction_result.get('key_features', {}), ensure_ascii=False)}
原始特征摘要：{json.dumps(prediction_result.get('raw_features', {}), ensure_ascii=False)}

请返回 JSON，字段必须包括：
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
                {"role": "system", "content": "你是网络安全分析助手，必须返回 JSON。"},
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
            "impact": "需要人工进一步确认。",
            "suggestion": "建议结合原始流量和日志再次核验。",
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
            f"模型将该流量判定为攻击流量，主要依据是多个关键特征出现明显异常，"
            f"其中包括 {', '.join(key_features.keys()) if key_features else '若干网络行为特征'}。"
        )
        impact = "该流量可能导致未授权访问、资源消耗异常或服务受影响。"
        suggestion = "建议立即核查来源IP、会话行为与目标主机状态，并结合防火墙或入侵防御策略进行限制。"
    else:
        risk_level = "Low"
        explanation = "模型将该流量判定为正常流量，当前未发现明显攻击特征。"
        impact = "短期内风险较低，但仍建议结合业务场景进行持续监测。"
        suggestion = "建议保留日志记录并继续观察，如后续同源流量异常增多可进一步分析。"

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
