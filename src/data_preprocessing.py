from __future__ import annotations

import json
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {file_path}")
    return pd.read_csv(file_path)


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [col for col in config.DROP_COLUMNS if col in df.columns]
    return df.drop(columns=existing) if existing else df


def split_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    if config.TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{config.TARGET_COLUMN}' not found in dataset.")

    y = df[config.TARGET_COLUMN].copy()
    attack_cat = df[config.AUX_LABEL_COLUMN].copy() if config.AUX_LABEL_COLUMN in df.columns else None

    feature_df = df.drop(columns=[config.TARGET_COLUMN], errors="ignore")
    if config.AUX_LABEL_COLUMN in feature_df.columns:
        feature_df = feature_df.drop(columns=[config.AUX_LABEL_COLUMN])

    return feature_df, y, attack_cat


def build_preprocessor(feature_df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = [col for col in feature_df.columns if col not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor, numerical_cols, categorical_cols


def transform_dataset(
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    if hasattr(x_train_transformed, "toarray"):
        x_train_transformed = x_train_transformed.toarray()
    if hasattr(x_test_transformed, "toarray"):
        x_test_transformed = x_test_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out().tolist()

    train_processed = pd.DataFrame(
        x_train_transformed,
        columns=feature_names,
        index=x_train.index,
    )
    test_processed = pd.DataFrame(
        x_test_transformed,
        columns=feature_names,
        index=x_test.index,
    )

    return train_processed, test_processed


def save_processed_data(
    x_train_processed: pd.DataFrame,
    x_test_processed: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_attack_cat: pd.Series | None,
    test_attack_cat: pd.Series | None,
) -> None:
    train_output = x_train_processed.copy()
    train_output[config.TARGET_COLUMN] = y_train.values
    if train_attack_cat is not None:
        train_output[config.AUX_LABEL_COLUMN] = train_attack_cat.values

    test_output = x_test_processed.copy()
    test_output[config.TARGET_COLUMN] = y_test.values
    if test_attack_cat is not None:
        test_output[config.AUX_LABEL_COLUMN] = test_attack_cat.values

    train_output.to_csv(config.TRAIN_PROCESSED_FILE, index=False)
    test_output.to_csv(config.TEST_PROCESSED_FILE, index=False)


def save_metadata(
    preprocessor: ColumnTransformer,
    feature_columns: list[str],
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> None:
    joblib.dump(preprocessor, config.PREPROCESSOR_FILE)
    joblib.dump(feature_columns, config.FEATURE_COLUMNS_FILE)

    metadata = {
        "target_column": config.TARGET_COLUMN,
        "aux_label_column": config.AUX_LABEL_COLUMN,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "processed_feature_count": len(feature_columns),
    }
    metadata_path = config.PROCESSED_DATA_DIR / "preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    config.ensure_directories()

    train_df = load_dataset(config.TRAIN_RAW_FILE)
    test_df = load_dataset(config.TEST_RAW_FILE)

    train_df = drop_unused_columns(train_df)
    test_df = drop_unused_columns(test_df)

    x_train, y_train, train_attack_cat = split_features_and_labels(train_df)
    x_test, y_test, test_attack_cat = split_features_and_labels(test_df)

    preprocessor, numerical_cols, categorical_cols = build_preprocessor(x_train)
    x_train_processed, x_test_processed = transform_dataset(preprocessor, x_train, x_test)

    save_processed_data(
        x_train_processed,
        x_test_processed,
        y_train,
        y_test,
        train_attack_cat,
        test_attack_cat,
    )
    save_metadata(
        preprocessor,
        x_train_processed.columns.tolist(),
        numerical_cols,
        categorical_cols,
    )

    print(f"Train raw shape: {train_df.shape}")
    print(f"Test raw shape: {test_df.shape}")
    print(f"Train processed shape: {x_train_processed.shape}")
    print(f"Test processed shape: {x_test_processed.shape}")
    print(f"Saved: {config.TRAIN_PROCESSED_FILE}")
    print(f"Saved: {config.TEST_PROCESSED_FILE}")


if __name__ == "__main__":
    main()
