from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.database import (
    export_results_to_csv,
    export_results_to_json,
    save_batch_results,
)
from src.llm_analysis import analyze_prediction_result


MODEL_PATHS = {
    "decision_tree": config.DECISION_TREE_MODEL_FILE,
    "random_forest": config.RANDOM_FOREST_MODEL_FILE,
    "svm": config.SVM_MODEL_FILE,
}

FEATURE_SUMMARY_COLUMNS = [
    "proto",
    "service",
    "state",
    "dur",
    "rate",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
]


def load_model(model_name: str | None = None):
    selected_name = model_name or get_default_model_name()
    return joblib.load(MODEL_PATHS[selected_name]), selected_name


def get_default_model_name() -> str:
    if config.BEST_MODEL_SUMMARY_FILE.exists():
        summary = json.loads(config.BEST_MODEL_SUMMARY_FILE.read_text(encoding="utf-8"))
        return summary.get("model_name", config.DEFAULT_MODEL_NAME)
    return config.DEFAULT_MODEL_NAME


def load_preprocessor():
    return joblib.load(config.PREPROCESSOR_FILE)


def load_feature_columns() -> list[str]:
    return joblib.load(config.FEATURE_COLUMNS_FILE)


def load_input_data(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def preprocess_input_data(
    df: pd.DataFrame,
    preprocessor,
    feature_columns: list[str],
) -> pd.DataFrame:
    model_input = df.drop(columns=[config.TARGET_COLUMN, config.AUX_LABEL_COLUMN], errors="ignore")
    model_input = model_input.drop(columns=config.DROP_COLUMNS, errors="ignore")
    transformed = preprocessor.transform(model_input)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed_df = pd.DataFrame(transformed, columns=feature_columns, index=df.index)
    return transformed_df


def build_key_features(raw_row: pd.Series, top_count: int = config.DEFAULT_TOP_FEATURE_COUNT) -> dict[str, Any]:
    picked: dict[str, Any] = {}
    for column in FEATURE_SUMMARY_COLUMNS:
        if column in raw_row.index:
            picked[column] = raw_row[column]
    return dict(list(picked.items())[:top_count])


def get_prediction_score(model, transformed_row: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(transformed_row)[0][1])
    elif hasattr(model, "decision_function"):
        raw_score = float(model.decision_function(transformed_row)[0])
        score = max(0.0, min(1.0, (raw_score + 1) / 2))
    else:
        score = 1.0
    return round(score, 6)


def format_prediction_result(
    raw_row: pd.Series,
    prediction_label: int,
    prediction_score: float,
    model_name: str,
    record_index: int,
) -> dict[str, Any]:
    record_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{record_index:04d}"
    prediction_text = "Attack" if int(prediction_label) == 1 else "Normal"
    return {
        "record_id": record_id,
        "model_name": model_name,
        "prediction_label": int(prediction_label),
        "prediction_text": prediction_text,
        "prediction_score": prediction_score,
        "raw_features": {
            key: raw_row[key]
            for key in raw_row.index
            if key not in {config.TARGET_COLUMN, config.AUX_LABEL_COLUMN}
        },
        "key_features": build_key_features(raw_row),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def enrich_with_analysis(prediction_result: dict[str, Any]) -> dict[str, Any]:
    analysis = analyze_prediction_result(prediction_result)
    merged = prediction_result.copy()
    merged.update(analysis)
    return merged


def sanitize_llm_options(llm_options: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not llm_options:
        return None
    cleaned = {
        "base_url": str(llm_options.get("base_url") or "").strip(),
        "model": str(llm_options.get("model") or "").strip(),
        "api_key": str(llm_options.get("api_key") or "").strip(),
    }
    if not any(cleaned.values()):
        return None
    return cleaned


def predict_batch(
    df: pd.DataFrame,
    model_name: str | None = None,
    save_results: bool = True,
    llm_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    model, selected_name = load_model(model_name)
    preprocessor = load_preprocessor()
    feature_columns = load_feature_columns()
    transformed_df = preprocess_input_data(df, preprocessor, feature_columns)
    predictions = model.predict(transformed_df)
    cleaned_llm_options = sanitize_llm_options(llm_options)

    results: list[dict[str, Any]] = []
    for idx, (row_index, raw_row) in enumerate(df.iterrows(), start=1):
        transformed_row = transformed_df.loc[[row_index]]
        score = get_prediction_score(model, transformed_row)
        prediction_result = format_prediction_result(
            raw_row=raw_row,
            prediction_label=int(predictions[idx - 1]),
            prediction_score=score,
            model_name=selected_name,
            record_index=idx,
        )
        analysis = analyze_prediction_result(prediction_result, llm_options=cleaned_llm_options)
        merged = prediction_result.copy()
        merged.update(analysis)
        results.append(merged)

    if save_results:
        save_batch_results(results)
        export_results_to_json(results)
        export_results_to_csv(results)
    return results


def predict_single(
    sample: dict[str, Any],
    model_name: str | None = None,
    save_result: bool = False,
    llm_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    df = pd.DataFrame([sample])
    result = predict_batch(
        df,
        model_name=model_name,
        save_results=save_result,
        llm_options=llm_options,
    )[0]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Network intrusion detection prediction tool.")
    parser.add_argument(
        "--input",
        default=str(config.TEST_RAW_FILE),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_PATHS.keys()),
        default=None,
        help="Model name to use. Defaults to the current best model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of rows to predict for quick testing.",
    )
    return parser.parse_args()


def main() -> None:
    config.ensure_directories()
    args = parse_args()
    input_df = load_input_data(Path(args.input))
    if args.limit:
        input_df = input_df.head(args.limit)
    results = predict_batch(input_df, model_name=args.model, save_results=True)
    print(json.dumps(results[:3], ensure_ascii=False, indent=2))
    print(f"Saved results -> {config.DETECTION_RESULTS_JSON_FILE}")
    print(f"Saved results -> {config.DETECTION_RESULTS_CSV_FILE}")


if __name__ == "__main__":
    main()
