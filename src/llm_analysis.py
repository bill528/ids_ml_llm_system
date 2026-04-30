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


def resolve_llm_options(llm_options: dict[str, Any] | None = None) -> dict[str, str]:
    options = llm_options or {}
    return {
        "api_key": str(options.get("api_key") or config.LLM_API_KEY or "").strip(),
        "base_url": str(options.get("base_url") or config.LLM_BASE_URL or "").strip(),
        "model": str(options.get("model") or config.LLM_MODEL or "").strip(),
    }


def build_prompt(prediction_result: dict[str, Any]) -> str:
    return f"""
你是一名网络安全分析助手。
请基于以下入侵检测结果进行辅助分析，并且仅返回 JSON 对象。
所有 explanation、impact、suggestion 字段都必须使用简体中文。

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


def call_llm_api(prompt: str, llm_options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = resolve_llm_options(llm_options)
    if not options["api_key"] or not options["model"] or not options["base_url"]:
        return {}

    response = requests.post(
        f"{options['base_url'].rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {options['api_key']}",
            "Content-Type": "application/json",
        },
        json={
            "model": options["model"],
            "messages": [
                {"role": "system", "content": "你是一名网络安全分析助手。只返回 JSON，并且 explanation、impact、suggestion 必须使用简体中文。"},
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
            "impact": "仍建议人工复核本次分析结果。",
            "suggestion": "在采取处置动作前，请先交叉核对原始流量与系统日志。",
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
            "模型将该流量判定为攻击，原因是多个关键特征表现异常，包括："
            f"{'、'.join(key_features.keys()) if key_features else '多个网络行为指标'}。"
        )
        impact = "该流量可能导致未授权访问、资源异常消耗或服务中断。"
        suggestion = "建议检查源 IP、会话行为和目标主机状态，并按需启用防火墙或入侵防护策略。"
    else:
        risk_level = "Low"
        explanation = "模型将该流量判定为正常，未发现明显攻击特征。"
        impact = "当前短期风险较低，但仍建议持续观察相关流量。"
        suggestion = "建议保留相关日志，并继续观察同类会话是否出现异常变化。"

    return {
        "risk_level": risk_level,
        "explanation": explanation,
        "impact": impact,
        "suggestion": suggestion,
    }


def analyze_prediction_result(
    prediction_result: dict[str, Any],
    llm_options: dict[str, Any] | None = None,
) -> dict[str, str]:
    prompt = build_prompt(prediction_result)
    try:
        response = call_llm_api(prompt, llm_options=llm_options)
        parsed = parse_llm_response(response)
        if parsed:
            return parsed
    except Exception:
        pass
    return fallback_analysis(prediction_result)
