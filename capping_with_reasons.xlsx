"""Train a small NumPy neural model for Transformer Health Index prediction."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from health_rules import (
    GAS_LABEL_TO_COLUMN,
    GAS_REASONS,
    RAW_FEATURES,
    GasRule,
    build_feature_row,
    feature_names,
    gas_conditions,
    gas_reason,
    gas_scores,
    gas_threshold_features,
    label_rule_features,
    label_rule_name,
    rule_summary,
    set_gas_reasons,
    set_gas_rules,
    validate_columns,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = APP_DIR / "data" / "DGA_HI_Smooth_Dataset_with_Reasons.xlsx"
MODEL_DIR = APP_DIR / "model"
OUTPUT_DIR = APP_DIR / "outputs"
AUGMENTED_DATASET = OUTPUT_DIR / "DGA_HI_Training_Set_Augmented.csv"


COLUMN_MAP = {
    "H2_ppm": "H2",
    "CH4_ppm": "Methane",
    "CH4": "Methane",
    "C2H6_ppm": "Ethane",
    "C2H6": "Ethane",
    "C2H4_ppm": "Ethylene",
    "C2H4": "Ethylene",
    "C2H2_ppm": "Acetylene",
    "C2H2": "Acetylene",
    "CO_ppm": "CO",
    "CO2_ppm": "CO2",
    "Overall_HI": "HI",
    "DGA_HI": "HI",
    "Smooth_HI": "HI",
}


def max_from_range(value: object) -> float:
    text = str(value).replace(",", "").strip()
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        raise ValueError(f"Could not parse threshold range: {value}")
    return max(numbers)


def configure_rules_from_workbook(dataset_path: Path) -> None:
    thresholds = pd.read_excel(dataset_path, sheet_name=1)
    rules: list[GasRule] = []
    reasons: dict[str, str] = {}
    for record in thresholds.to_dict("records"):
        gas = str(record["Gas"]).strip().replace("₂", "2").replace("₄", "4").replace("₆", "6")
        column = GAS_LABEL_TO_COLUMN.get(gas)
        if not column:
            continue
        healthy_max = float(record["Healthy Max"]) if "Healthy Max" in record else max_from_range(record["Healthy"])
        moderate_max = float(record["Moderate Max"]) if "Moderate Max" in record else max_from_range(record["Moderate"])
        poor_max = float(record["Poor Max"]) if "Poor Max" in record else max_from_range(record["Poor"])
        rules.append(
            GasRule(
                column=column,
                label=gas,
                healthy_max=healthy_max,
                moderate_max=moderate_max,
                poor_max=poor_max,
            )
        )
        if "High Value Reason" in record:
            reasons[column] = str(record["High Value Reason"])
    if len(rules) != 7:
        raise ValueError("Thresholds sheet must contain 7 gas rows with Gas and threshold ranges")
    set_gas_rules(rules)
    if reasons:
        set_gas_reasons(reasons)


def read_training_sheet(dataset_path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(dataset_path)
    for sheet_name in ("Training_Data", "Training_Set"):
        if sheet_name in xl.sheet_names:
            return pd.read_excel(dataset_path, sheet_name=sheet_name)
    return pd.read_excel(dataset_path, sheet_name=0)


def hi_as_points(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if float(numeric.max()) <= 1.5:
        return numeric * 100.0
    return numeric


def load_training_data(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    configure_rules_from_workbook(dataset_path)
    df = read_training_sheet(dataset_path)
    available_map = {source: target for source, target in COLUMN_MAP.items() if source in df.columns}
    df = df.rename(columns=available_map)
    validate_columns(df.columns)
    df = df.dropna(subset=["HI"]).copy()
    for column in RAW_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=RAW_FEATURES)

    hi_points = hi_as_points(df["HI"])
    critical_flags = []
    for row in df.to_dict("records"):
        scores = gas_scores(row)
        critical_flags.append(label_rule_features(scores)["label_rule_any_gas_critical"])
    df["label_rule_any_gas_critical"] = critical_flags
    df["HI_before_label_rule"] = hi_points
    df["HI_training_target"] = np.where(df["label_rule_any_gas_critical"] == 1.0, 0.0, hi_points)
    x = np.array([build_feature_row(row) for row in df.to_dict("records")], dtype=np.float64)
    y = df["HI_training_target"].to_numpy(dtype=np.float64).reshape(-1, 1)
    return x, y, df


def build_augmented_dataset(df: pd.DataFrame) -> pd.DataFrame:
    augmented = df.copy()
    for row_index, row in enumerate(df.to_dict("records")):
        scores = gas_scores(row)
        conditions = gas_conditions(row)
        summary = rule_summary(scores)
        thresholds = gas_threshold_features(row)
        label_features = label_rule_features(scores)

        for key, value in scores.items():
            augmented.loc[row_index, key] = value
        for gas, condition in conditions.items():
            augmented.loc[row_index, f"{gas}_condition_from_threshold_table"] = condition
            if condition != "Healthy":
                augmented.loc[row_index, f"{gas}_reason_from_threshold_table"] = gas_reason(gas)
        for key, value in summary.items():
            augmented.loc[row_index, key] = value
        for key, value in label_features.items():
            augmented.loc[row_index, key] = value
        augmented.loc[row_index, "Label_Rule_Derived"] = label_rule_name(scores)
        for key, value in thresholds.items():
            augmented.loc[row_index, key] = value

    augmented["DGA_threshold_table_used_for_training"] = 1
    augmented["DGA_threshold_table_version"] = "Healthy/Moderate/Poor/Critical score table"
    return augmented


def split_data(x: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(len(x))
    test_size = max(1, int(len(x) * 0.2))
    test_idx = indexes[:test_size]
    train_idx = indexes[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def relu(value: np.ndarray) -> np.ndarray:
    return np.maximum(value, 0.0)


def train_network(x_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_scaled = y_train / 100.0
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_scaled = (x_train - mean) / std

    hidden = 28
    w1 = rng.normal(0, np.sqrt(2 / x_scaled.shape[1]), size=(x_scaled.shape[1], hidden))
    b1 = np.zeros((1, hidden))
    w2 = rng.normal(0, np.sqrt(2 / hidden), size=(hidden, 1))
    b2 = np.zeros((1, 1))

    mw1 = np.zeros_like(w1)
    vw1 = np.zeros_like(w1)
    mb1 = np.zeros_like(b1)
    vb1 = np.zeros_like(b1)
    mw2 = np.zeros_like(w2)
    vw2 = np.zeros_like(w2)
    mb2 = np.zeros_like(b2)
    vb2 = np.zeros_like(b2)

    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    l2 = 0.0008
    n = x_scaled.shape[0]

    for step in range(1, 3501):
        z1 = x_scaled @ w1 + b1
        a1 = relu(z1)
        prediction = a1 @ w2 + b2
        error = prediction - y_scaled

        d_pred = (2.0 / n) * error
        dw2 = a1.T @ d_pred + l2 * w2
        db2 = d_pred.sum(axis=0, keepdims=True)
        da1 = d_pred @ w2.T
        dz1 = da1 * (z1 > 0)
        dw1 = x_scaled.T @ dz1 + l2 * w1
        db1 = dz1.sum(axis=0, keepdims=True)

        for param, grad, mom, vel in (
            (w1, dw1, mw1, vw1),
            (b1, db1, mb1, vb1),
            (w2, dw2, mw2, vw2),
            (b2, db2, mb2, vb2),
        ):
            mom *= beta1
            mom += (1 - beta1) * grad
            vel *= beta2
            vel += (1 - beta2) * (grad * grad)
            m_hat = mom / (1 - beta1**step)
            v_hat = vel / (1 - beta2**step)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        if step in (1600, 2600):
            learning_rate *= 0.45

    return {"mean": mean, "std": std, "w1": w1, "b1": b1, "w2": w2, "b2": b2}


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    x_scaled = (x - model["mean"]) / model["std"]
    values = relu(x_scaled @ model["w1"] + model["b1"]) @ model["w2"] + model["b2"]
    return np.clip(values * 100.0, 0.0, 100.0)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted.reshape(-1) - actual.reshape(-1)
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((actual.reshape(-1) - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    x, y, df = load_training_data(args.dataset)
    augmented_df = build_augmented_dataset(df)
    x_train, x_test, y_train, y_test = split_data(x, y)
    model = train_network(x_train, y_train)
    train_pred = predict(model, x_train)
    test_pred = predict(model, x_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MODEL_DIR / "health_index_model.npz", **model)
    augmented_df.to_csv(AUGMENTED_DATASET, index=False)

    metadata = {
        "dataset": str(args.dataset.relative_to(APP_DIR)) if args.dataset.is_relative_to(APP_DIR) else str(args.dataset),
        "augmented_dataset": str(AUGMENTED_DATASET.relative_to(APP_DIR)) if AUGMENTED_DATASET.is_relative_to(APP_DIR) else str(AUGMENTED_DATASET),
        "rows_used": int(len(df)),
        "features": feature_names(),
        "target": "HI_training_target",
        "target_rule": "HI_training_target = 0 when any gas value crosses Poor Max from workbook Thresholds tab; otherwise source HI * 100",
        "train_metrics": metrics(y_train, train_pred),
        "test_metrics": metrics(y_test, test_pred),
        "sample_prediction": float(test_pred[0][0]),
        "sample_actual": float(y_test[0][0]),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
