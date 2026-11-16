"""Load the trained MLP and predict marks from prepared feature rows."""
from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np


def load_trained_model(model_path="nn/model.joblib"):
    artifact = joblib.load(model_path)
    return artifact["model"], artifact["features"]


def predict_student_assignments(
    data_path="data/processed/fff_tma_examples.csv",
    model_path="nn/model.joblib",
    presentation="2014J",
):
    model, features = load_trained_model(model_path)
    with Path(data_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    # Use one held-out student with observed examples for the first three TMAs.
    students = {}
    for row in rows:
        if row["presentation"] != presentation:
            continue
        students.setdefault(row["student_id"], {})[int(row["assessment_number"])] = row

    selected = next(
        (items for items in students.values() if all(i in items for i in (1, 2, 3))),
        None,
    )
    if selected is None:
        raise RuntimeError("No held-out student has examples for the first three TMAs.")

    chosen = [selected[i] for i in (1, 2, 3)]
    matrix = np.asarray([
        [float(row[name]) if row[name] != "" else np.nan for name in features]
        for row in chosen
    ], dtype=float)
    predictions = np.clip(model.predict(matrix), 0.0, 100.0)
    return tuple(float(x) for x in predictions), chosen[0]["student_id"]
