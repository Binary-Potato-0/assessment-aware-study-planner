"""Load the trained model and predict marks for a held-out student."""
from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np


# MODEL LOADING: restore the trained pipeline and its feature order.
def load_trained_model(model_path="nn/model.joblib"):
    """Load the model and the feature order saved during training."""
    saved_artifact = joblib.load(model_path)
    return saved_artifact["model"], saved_artifact["features"]


# STUDENT PREDICTION: select one held-out student and predict their first 3 TMAs.
def predict_student_assignments(
    data_path="data/processed/fff_tma_examples.csv",
    model_path="nn/model.joblib",
    presentation="2014J",
    target_mark=65.0,
):
    """Predict the first three assessments for a useful demo student."""
    model, feature_names = load_trained_model(model_path)

    with Path(data_path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    # FILTER + GROUP: keep records from the requested presentation, grouped by student.
    # Group assessment records by student, using only the held-out presentation.
    students = {}
    for row in rows:
        if row["presentation"] != presentation:
            continue

        student_id = row["student_id"]
        assessment_number = int(row["assessment_number"])
        students.setdefault(student_id, {})[assessment_number] = row

    # PREDICT CANDIDATES: run the model for students with all three assessment rows.
    candidate_predictions = []

    for student_id, assessments in students.items():
        if not all(number in assessments for number in (1, 2, 3)):
            continue

        selected_rows = [assessments[number] for number in (1, 2, 3)]
        feature_matrix = np.asarray([
            [
                float(row[name]) if row[name] != "" else np.nan
                for name in feature_names
            ]
            for row in selected_rows
        ])

        predictions = np.clip(model.predict(feature_matrix), 0, 100)
        candidate_predictions.append(
            (tuple(float(mark) for mark in predictions), student_id)
        )

    if not candidate_predictions:
        raise RuntimeError("No held-out student has records for assessments 1, 2 and 3.")

    # DEMO STUDENT SELECTION: prefer a student below target to show a non-zero plan.
    # Prefer a student below target in all three assessments, so the demo
    # demonstrates a non-zero allocation. Selection uses predictions only.
    below_target = [
        candidate for candidate in candidate_predictions
        if all(mark < target_mark for mark in candidate[0])
    ]

    if below_target:
        chosen_marks, chosen_id = min(
            below_target,
            key=lambda candidate: sum(target_mark - mark for mark in candidate[0]),
        )
    else:
        chosen_marks, chosen_id = max(
            candidate_predictions,
            key=lambda candidate: sum(
                mark < target_mark for mark in candidate[0]
            ),
        )

    return chosen_marks, chosen_id
