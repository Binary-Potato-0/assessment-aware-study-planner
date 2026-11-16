"""Load the trained MLP and select a useful held-out student for the demo."""
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
    target_mark=65.0,
):
    """Predict the first three TMA marks for a held-out student.

    Prefer a student whose three predicted marks are all below the demo target,
    choosing the one closest to the target. This makes the integrated demo
    exercise Search and CSP instead of immediately returning zero sessions.
    """
    model, features = load_trained_model(model_path)
    with Path(data_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    students = {}
    for row in rows:
        if row["presentation"] != presentation:
            continue
        students.setdefault(row["student_id"], {})[int(row["assessment_number"])] = row

    candidates = []
    for student_id, items in students.items():
        if not all(i in items for i in (1, 2, 3)):
            continue
        chosen = [items[i] for i in (1, 2, 3)]
        matrix = np.asarray([
            [float(row[name]) if row[name] != "" else np.nan for name in features]
            for row in chosen
        ], dtype=float)
        predictions = np.clip(model.predict(matrix), 0.0, 100.0)
        marks = tuple(float(x) for x in predictions)
        candidates.append((marks, student_id))

    if not candidates:
        raise RuntimeError("No held-out student has examples for the first three TMAs.")

    below_target = [
        item for item in candidates
        if all(mark < target_mark for mark in item[0])
    ]
    if below_target:
        # Closest all-below-target student gives a non-trivial, realistic demo.
        selected_marks, selected_id = min(
            below_target,
            key=lambda item: sum(target_mark - mark for mark in item[0]),
        )
    else:
        # Fallback: choose the student with the greatest number of assessments
        # below target, then the largest total deficit.
        selected_marks, selected_id = max(
            candidates,
            key=lambda item: (
                sum(mark < target_mark for mark in item[0]),
                sum(max(0.0, target_mark - mark) for mark in item[0]),
            ),
        )

    return selected_marks, selected_id
