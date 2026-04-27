from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
SCALERS_DIR = MODELS_DIR / "scalers"
ENCODERS_DIR = MODELS_DIR / "encoders"

DATABASE_DIR = BASE_DIR / "database"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
EXPORTS_DIR = REPORTS_DIR / "exports"

TRAIN_RAW_FILE = RAW_DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_RAW_FILE = RAW_DATA_DIR / "UNSW_NB15_testing-set.csv"

TRAIN_PROCESSED_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PROCESSED_FILE = PROCESSED_DATA_DIR / "test_processed.csv"

PREPROCESSOR_FILE = SCALERS_DIR / "preprocessor.joblib"
FEATURE_COLUMNS_FILE = ENCODERS_DIR / "feature_columns.joblib"

DATABASE_FILE = DATABASE_DIR / "ids.db"

DECISION_TREE_MODEL_FILE = SAVED_MODELS_DIR / "decision_tree_model.joblib"
RANDOM_FOREST_MODEL_FILE = SAVED_MODELS_DIR / "random_forest_model.joblib"
SVM_MODEL_FILE = SAVED_MODELS_DIR / "svm_model.joblib"

MODEL_RESULTS_FILE = METRICS_DIR / "model_metrics.csv"
MODEL_RESULTS_JSON_FILE = METRICS_DIR / "model_metrics.json"
BEST_MODEL_SUMMARY_FILE = METRICS_DIR / "best_model_summary.json"

ACCURACY_FIGURE_FILE = FIGURES_DIR / "model_accuracy_comparison.png"
RECALL_FIGURE_FILE = FIGURES_DIR / "model_recall_comparison.png"
F1_FIGURE_FILE = FIGURES_DIR / "model_f1_comparison.png"
BEST_MODEL_CONFUSION_FIGURE_FILE = FIGURES_DIR / "best_model_confusion_matrix.png"

TARGET_COLUMN = "label"
AUX_LABEL_COLUMN = "attack_cat"
DROP_COLUMNS = ["id"]


def ensure_directories() -> None:
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        SAVED_MODELS_DIR,
        SCALERS_DIR,
        ENCODERS_DIR,
        DATABASE_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        EXPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
