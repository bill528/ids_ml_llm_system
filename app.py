from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import config
from src.database import (
    count_detection_history,
    initialize_database,
    query_detection_history,
    summarize_detection_history,
)
from src.predict import load_input_data, predict_batch, predict_single


app = Flask(__name__)

ALLOWED_MODELS = {"decision_tree", "random_forest", "svm"}


def json_error(message: str, status_code: int = 400, details: dict | None = None):
    payload = {"success": False, "error": message}
    if details:
        payload["details"] = details
    return jsonify(payload), status_code


def parse_positive_int(value, field_name: str, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer")
    if parsed < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0")
    return parsed


def validate_model_name(model_name: str | None) -> str | None:
    if not model_name:
        return None
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"model_name must be one of: {', '.join(sorted(ALLOWED_MODELS))}")
    return model_name


def normalize_datetime(value: str | None, field_name: str, end_of_day: bool = False) -> str | None:
    if not value:
        return None
    raw = value.strip()
    patterns = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
    for pattern in patterns:
        try:
            parsed = datetime.strptime(raw, pattern)
            if pattern == "%Y-%m-%d" and end_of_day:
                parsed = parsed.replace(hour=23, minute=59, second=59)
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    raise ValueError(f"{field_name} must be in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format")


@app.get("/")
def index():
    return render_template("dashboard.html", default_csv_path=str(config.TEST_RAW_FILE))


@app.get("/api")
def api_index():
    return jsonify(
        {
            "project": "ids_ml_llm_system",
            "status": "running",
            "message": "Phase three API is available.",
            "endpoints": {
                "health": "/api/health",
                "detect_csv": "/api/detect/csv",
                "detect_single": "/api/detect/single",
                "history": "/api/history",
                "history_summary": "/api/history/summary",
            },
        }
    )


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/detect/csv")
def detect_csv():
    payload = request.get_json(silent=True) or {}
    input_path = payload.get("input_path")
    try:
        model_name = validate_model_name(payload.get("model_name"))
        limit = parse_positive_int(payload.get("limit"), "limit", default=10)
    except ValueError as exc:
        return json_error(str(exc))

    if not input_path:
        return json_error("input_path is required")

    csv_path = Path(input_path)
    if not csv_path.exists():
        return json_error(f"input file not found: {csv_path}", status_code=404)
    if csv_path.suffix.lower() != ".csv":
        return json_error("input_path must point to a CSV file")

    try:
        input_df = load_input_data(csv_path)
        if input_df.empty:
            return json_error("input CSV is empty")
        if limit:
            input_df = input_df.head(limit)
        results = predict_batch(input_df, model_name=model_name, save_results=True)
    except Exception as exc:
        return json_error("failed to run batch detection", details={"message": str(exc)}, status_code=500)

    summary = build_result_summary(results)
    return jsonify(
        {
            "success": True,
            "count": len(results),
            "model_name": results[0]["model_name"] if results else model_name,
            "summary": summary,
            "results": results,
        }
    )


@app.post("/api/detect/single")
def detect_single():
    payload = request.get_json(silent=True) or {}
    sample = payload.get("sample")
    try:
        model_name = validate_model_name(payload.get("model_name"))
    except ValueError as exc:
        return json_error(str(exc))

    if not sample or not isinstance(sample, dict):
        return json_error("sample must be a JSON object")

    try:
        result = predict_single(sample=sample, model_name=model_name, save_result=True)
    except Exception as exc:
        return json_error("failed to run single detection", details={"message": str(exc)}, status_code=500)
    return jsonify({"success": True, "result": result})


@app.get("/api/history")
def history():
    try:
        limit = parse_positive_int(request.args.get("limit"), "limit", default=20)
        offset = parse_positive_int(request.args.get("offset"), "offset", default=0) or 0
        model_name = validate_model_name(request.args.get("model_name"))
        risk_level = request.args.get("risk_level") or None
        prediction_text = request.args.get("prediction_text") or None
        created_from = normalize_datetime(request.args.get("created_from"), "created_from")
        created_to = normalize_datetime(request.args.get("created_to"), "created_to", end_of_day=True)
    except ValueError as exc:
        return json_error(str(exc))

    results = query_detection_history(
        limit=limit,
        offset=offset,
        model_name=model_name,
        risk_level=risk_level,
        prediction_text=prediction_text,
        created_from=created_from,
        created_to=created_to,
    )
    total = count_detection_history(
        model_name=model_name,
        risk_level=risk_level,
        prediction_text=prediction_text,
        created_from=created_from,
        created_to=created_to,
    )
    return jsonify(
        {
            "success": True,
            "count": len(results),
            "total": total,
            "limit": limit,
            "offset": offset,
            "results": results,
        }
    )


@app.get("/api/history/summary")
def history_summary():
    try:
        model_name = validate_model_name(request.args.get("model_name"))
        risk_level = request.args.get("risk_level") or None
        prediction_text = request.args.get("prediction_text") or None
        created_from = normalize_datetime(request.args.get("created_from"), "created_from")
        created_to = normalize_datetime(request.args.get("created_to"), "created_to", end_of_day=True)
    except ValueError as exc:
        return json_error(str(exc))
    return jsonify(
        {
            "success": True,
            **summarize_detection_history(
                model_name=model_name,
                risk_level=risk_level,
                prediction_text=prediction_text,
                created_from=created_from,
                created_to=created_to,
            ),
        }
    )


@app.errorhandler(404)
def handle_not_found(_error):
    if request.path.startswith("/api/"):
        return json_error("endpoint not found", status_code=404)
    return render_template("dashboard.html"), 404


@app.errorhandler(500)
def handle_server_error(_error):
    if request.path.startswith("/api/"):
        return json_error("internal server error", status_code=500)
    return render_template("dashboard.html"), 500


def build_result_summary(results: list[dict]) -> dict:
    summary = {
        "total": len(results),
        "prediction_counts": {},
        "risk_counts": {},
    }
    for item in results:
        summary["prediction_counts"][item["prediction_text"]] = summary["prediction_counts"].get(item["prediction_text"], 0) + 1
        summary["risk_counts"][item["risk_level"]] = summary["risk_counts"].get(item["risk_level"], 0) + 1
    return summary


def create_app() -> Flask:
    config.ensure_directories()
    initialize_database()
    return app


if __name__ == "__main__":
    create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
