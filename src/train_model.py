from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def load_processed_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(config.TRAIN_PROCESSED_FILE)
    test_df = pd.read_csv(config.TEST_PROCESSED_FILE)

    drop_columns = [config.TARGET_COLUMN]
    if config.AUX_LABEL_COLUMN in train_df.columns:
        drop_columns.append(config.AUX_LABEL_COLUMN)

    x_train = train_df.drop(columns=drop_columns, errors="ignore")
    y_train = train_df[config.TARGET_COLUMN]
    x_test = test_df.drop(columns=drop_columns, errors="ignore")
    y_test = test_df[config.TARGET_COLUMN]
    return x_train, y_train, x_test, y_test


def build_models() -> dict[str, object]:
    return {
        "decision_tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=42,
        ),
        # LinearSVC is a practical SVM implementation for this dataset size.
        "svm": LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42,
        ),
    }


def model_output_path(model_name: str) -> Path:
    mapping = {
        "decision_tree": config.DECISION_TREE_MODEL_FILE,
        "random_forest": config.RANDOM_FOREST_MODEL_FILE,
        "svm": config.SVM_MODEL_FILE,
    }
    return mapping[model_name]


def train_and_save_models(
    models: dict[str, object],
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> list[dict[str, object]]:
    training_summary: list[dict[str, object]] = []

    for model_name, model in models.items():
        start_time = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - start_time

        output_path = model_output_path(model_name)
        joblib.dump(model, output_path)

        training_summary.append(
            {
                "model_name": model_name,
                "train_seconds": round(train_seconds, 4),
                "output_path": str(output_path),
            }
        )
        print(f"Trained {model_name} in {train_seconds:.4f}s -> {output_path}")

    return training_summary


def save_training_summary(summary: list[dict[str, object]]) -> None:
    summary_path = config.METRICS_DIR / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved training summary -> {summary_path}")


def main() -> None:
    config.ensure_directories()
    x_train, y_train, x_test, y_test = load_processed_data()
    print(f"Training feature shape: {x_train.shape}")
    print(f"Test feature shape: {x_test.shape}")
    print(f"Training labels: {y_train.value_counts().to_dict()}")
    print(f"Test labels: {y_test.value_counts().to_dict()}")

    models = build_models()
    training_summary = train_and_save_models(models, x_train, y_train)
    save_training_summary(training_summary)


if __name__ == "__main__":
    main()
