from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATASET_PATH = DATA_DIR / "winequality-red.csv"
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

STUDENT_NAME = "SriHarsha Bodicherla"
ROLL_NO = "2022BCD0002"


# Edit this block manually for each experiment.
EXPERIMENT_CONFIG = {
    "model_name": "lasso",
    "test_size": 0.2,
    "random_state": 42,
    "scaler": "standard",  # options: none, standard, minmax
    "feature_selection": "k_best",  # options: none, k_best
    "k_best_features": 8,
    "model_params": {"alpha": 0.1},
}


def ensure_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        print(f"Dataset not found. Downloading from {DATASET_URL}...")
        urlretrieve(DATASET_URL, DATASET_PATH)
    return DATASET_PATH


def build_model(name: str, params: dict) -> object:
    models = {
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
    }
    if name not in models:
        valid = ", ".join(sorted(models))
        raise ValueError(f"Unsupported model '{name}'. Choose from: {valid}")
    return models[name](**params)


def build_pipeline(config: dict) -> Pipeline:
    steps: list[tuple[str, object]] = []

    scaler_name = config["scaler"]
    if scaler_name == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler_name == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif scaler_name != "none":
        raise ValueError("Scaler must be one of: none, standard, minmax")

    feature_selection = config["feature_selection"]
    if feature_selection == "k_best":
        steps.append(
            (
                "feature_selection",
                SelectKBest(score_func=f_regression, k=config["k_best_features"]),
            )
        )
    elif feature_selection != "none":
        raise ValueError("Feature selection must be one of: none, k_best")

    steps.append(
        ("model", build_model(config["model_name"], config.get("model_params", {})))
    )
    return Pipeline(steps)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = ensure_dataset()

    dataframe = pd.read_csv(dataset_path, sep=";")
    X = dataframe.drop(columns=["quality"])
    y = dataframe["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=EXPERIMENT_CONFIG["test_size"],
        random_state=EXPERIMENT_CONFIG["random_state"],
    )

    pipeline = build_pipeline(EXPERIMENT_CONFIG)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    model_path = OUTPUT_DIR / "trained_model.pkl"
    metrics_path = OUTPUT_DIR / "results.json"

    joblib.dump(pipeline, model_path)

    payload = {
        "student_name": STUDENT_NAME,
        "roll_no": ROLL_NO,
        "dataset": dataset_path.name,
        "model_name": EXPERIMENT_CONFIG["model_name"],
        "test_size": EXPERIMENT_CONFIG["test_size"],
        "random_state": EXPERIMENT_CONFIG["random_state"],
        "scaler": EXPERIMENT_CONFIG["scaler"],
        "feature_selection": EXPERIMENT_CONFIG["feature_selection"],
        "k_best_features": EXPERIMENT_CONFIG["k_best_features"],
        "model_params": EXPERIMENT_CONFIG["model_params"],
        "mse": mse,
        "r2_score": r2,
        "model_path": str(model_path),
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"Student Name: {STUDENT_NAME}")
    print(f"Roll Number: {ROLL_NO}")
    print(f"Model: {EXPERIMENT_CONFIG['model_name']}")
    print(f"MSE: {mse:.6f}")
    print(f"R2 Score: {r2:.6f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved results to: {metrics_path}")


if __name__ == "__main__":
    main()

