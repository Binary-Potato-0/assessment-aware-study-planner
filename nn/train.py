"""Train and evaluate an MLP against a student-history baseline.

The split is presentation-based: FFF 2013J trains; FFF 2014J tests.
The model predicts observed marks from prior marks and pre-deadline activity.
It does not estimate the causal effect of additional study sessions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "assessment_number",
    "assessment_day",
    "assessment_weight",
    "previous_score_count",
    "previous_score_mean",
    "previous_score_last",
    "clicks_preceding_28d",
    "clicks_preceding_7d",
]
PREVIOUS_MEAN_INDEX = FEATURES.index("previous_score_mean")


def load_split(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    if not train_rows or not test_rows:
        raise ValueError(
            f"Expected non-empty train and test splits; got {len(train_rows)} and {len(test_rows)}"
        )

    def arrays(part):
        x = np.array(
            [[float(r[name]) if r[name] != "" else np.nan for name in FEATURES] for r in part],
            dtype=float,
        )
        y = np.array([float(r["target_score"]) for r in part], dtype=float)
        return x, y

    return (*arrays(train_rows), *arrays(test_rows))


def metrics(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/fff_tma_examples.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("nn/results.json"))
    parser.add_argument("--model-output", type=Path, default=Path("nn/model.joblib"))
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_split(args.data)

    # Stronger and more relevant than a global mean: predict each student's
    # mean of scores actually available before the target assessment deadline.
    # Students without prior scores fall back to the training-set mean.
    prior_means = x_test[:, PREVIOUS_MEAN_INDEX]
    baseline_predictions = np.where(
        np.isfinite(prior_means), prior_means, float(np.mean(y_train))
    )
    baseline_predictions = np.clip(baseline_predictions, 0.0, 100.0)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])
    model.fit(x_train, y_train)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, args.model_output)
    predictions = np.clip(model.predict(x_test), 0.0, 100.0)

    result = {
        "dataset": str(args.data),
        "saved_model": str(args.model_output),
        "train_presentation": "FFF 2013J",
        "test_presentation": "FFF 2014J",
        "train_examples": int(len(y_train)),
        "test_examples": int(len(y_test)),
        "features": FEATURES,
        "baseline_student_previous_score_mean": metrics(y_test, baseline_predictions),
        "mlp": metrics(y_test, predictions),
        "mlp_iterations": int(model.named_steps["mlp"].n_iter_),
        "interpretation": (
            "Predictive evaluation across presentations. The baseline uses each "
            "student's mean of scores submitted before the target deadline, falling "
            "back to the training mean when no prior score exists. This does not "
            "establish that study sessions or clicks cause score improvements."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
