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
