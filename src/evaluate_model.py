from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

MODEL_LABELS = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "SVM",
}


def load_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    test_df = pd.read_csv(config.TEST_PROCESSED_FILE)
    drop_columns = [config.TARGET_COLUMN]
    if config.AUX_LABEL_COLUMN in test_df.columns:
        drop_columns.append(config.AUX_LABEL_COLUMN)

    x_test = test_df.drop(columns=drop_columns, errors="ignore")
    y_test = test_df[config.TARGET_COLUMN]
    return x_test, y_test


def load_models() -> dict[str, object]:
    return {
        "decision_tree": joblib.load(config.DECISION_TREE_MODEL_FILE),
        "random_forest": joblib.load(config.RANDOM_FOREST_MODEL_FILE),
        "svm": joblib.load(config.SVM_MODEL_FILE),
    }


def evaluate_models(models: dict[str, object], x_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, dict[str, list[list[int]]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    confusion_matrices: dict[str, list[list[int]]] = {}
    reports: dict[str, dict[str, object]] = {}

    for model_name, model in models.items():
        start_time = time.perf_counter()
        predictions = model.predict(x_test)
        predict_seconds = time.perf_counter() - start_time

        cm = confusion_matrix(y_test, predictions)
        confusion_matrices[model_name] = cm.tolist()
        reports[model_name] = classification_report(y_test, predictions, output_dict=True)

        rows.append(
            {
                "model_name": model_name,
                "accuracy": round(accuracy_score(y_test, predictions), 6),
                "precision": round(precision_score(y_test, predictions, zero_division=0), 6),
                "recall": round(recall_score(y_test, predictions, zero_division=0), 6),
                "f1_score": round(f1_score(y_test, predictions, zero_division=0), 6),
                "predict_seconds": round(predict_seconds, 6),
            }
        )

    result_df = pd.DataFrame(rows).sort_values(by=["f1_score", "recall", "accuracy"], ascending=False)
    return result_df, confusion_matrices, reports


def save_metrics(result_df: pd.DataFrame, confusion_matrices: dict[str, list[list[int]]], reports: dict[str, dict[str, object]]) -> dict[str, object]:
    result_df.to_csv(config.MODEL_RESULTS_FILE, index=False)
    metrics_json = {
        "model_metrics": result_df.to_dict(orient="records"),
        "confusion_matrices": confusion_matrices,
        "classification_reports": reports,
    }
    config.MODEL_RESULTS_JSON_FILE.write_text(
        json.dumps(metrics_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_model = result_df.iloc[0].to_dict()
    config.BEST_MODEL_SUMMARY_FILE.write_text(
        json.dumps(best_model, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return best_model


def create_bar_chart(result_df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    sns.set_theme(style="whitegrid")
    plot_df = result_df.copy()
    plot_df["display_name"] = plot_df["model_name"].map(MODEL_LABELS)
    plt.figure(figsize=(9, 5.6))
    ax = sns.barplot(
        data=plot_df,
        x="display_name",
        y=metric,
        hue="display_name",
        palette=["#6baed6", "#4292c6", "#2171b5"],
        legend=False,
        width=0.62,
    )
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylim(0, 1.06)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            min(height + 0.015, 1.03),
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_confusion_matrix_figure(best_model_name: str, confusion_matrices: dict[str, list[list[int]]]) -> None:
    matrix = confusion_matrices[best_model_name]
    sns.set_theme(style="white")
    plt.figure(figsize=(7, 5.8))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        annot_kws={"size": 12},
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
    )
    ax.set_title(f"Confusion Matrix - {MODEL_LABELS[best_model_name]}", fontsize=16, pad=14)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(config.BEST_MODEL_CONFUSION_FIGURE_FILE, dpi=300)
    plt.close()


def main() -> None:
    config.ensure_directories()
    x_test, y_test = load_processed_data()
    models = load_models()

    result_df, confusion_matrices, reports = evaluate_models(models, x_test, y_test)
    best_model = save_metrics(result_df, confusion_matrices, reports)

    create_bar_chart(result_df, "accuracy", config.ACCURACY_FIGURE_FILE, "Model Accuracy Comparison")
    create_bar_chart(result_df, "recall", config.RECALL_FIGURE_FILE, "Model Recall Comparison")
    create_bar_chart(result_df, "f1_score", config.F1_FIGURE_FILE, "Model F1 Score Comparison")
    create_confusion_matrix_figure(best_model["model_name"], confusion_matrices)

    print(result_df.to_string(index=False))
    print(f"Best model -> {best_model['model_name']}")
    print(f"Saved metrics -> {config.MODEL_RESULTS_FILE}")
    print(f"Saved figures -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
