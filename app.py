from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request

import config
from src.database import initialize_database, load_detection_history
from src.predict import load_input_data, predict_batch, predict_single


app = Flask(__name__)


@app.get("/")
def index():
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
    model_name = payload.get("model_name")
    limit = payload.get("limit")

    if not input_path:
        return jsonify({"error": "input_path is required"}), 400

    csv_path = Path(input_path)
    if not csv_path.exists():
        return jsonify({"error": f"input file not found: {csv_path}"}), 404

    input_df = load_input_data(csv_path)
    if limit:
        input_df = input_df.head(int(limit))

    results = predict_batch(input_df, model_name=model_name, save_results=True)
    return jsonify(
        {
            "count": len(results),
            "model_name": results[0]["model_name"] if results else model_name,
            "results": results,
        }
    )


@app.post("/api/detect/single")
def detect_single():
    payload = request.get_json(silent=True) or {}
    sample = payload.get("sample")
    model_name = payload.get("model_name")

    if not sample or not isinstance(sample, dict):
        return jsonify({"error": "sample must be a JSON object"}), 400

    result = predict_single(sample=sample, model_name=model_name, save_result=True)
    return jsonify(result)


@app.get("/api/history")
def history():
    limit = request.args.get("limit", default=20, type=int)
    results = load_detection_history(limit=limit)
    return jsonify({"count": len(results), "results": results})


def create_app() -> Flask:
    config.ensure_directories()
    initialize_database()
    return app


if __name__ == "__main__":
    create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
