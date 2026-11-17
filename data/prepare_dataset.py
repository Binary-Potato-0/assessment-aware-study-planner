"""Build leakage-aware student/assessment examples from local OULAD CSV files.

Only scores from earlier assessments that were actually submitted before the
current deadline are used as history. Clicks are aggregated strictly before
the deadline. This is predictive association, not causal evidence.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

# ============================================================
# DATASET CONFIGURATION: module, presentations, and activity windows
# ============================================================
MODULE = "FFF"
PRESENTATIONS = ("2013J", "2014J")
TRAIN_PRESENTATION = "2013J"
WINDOW_DAYS = 28
RECENT_DAYS = 7


# CSV READER: read OULAD files while handling their UTF-8 BOM.
def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


# ============================================================
# DATASET BUILDER: raw OULAD tables -> one row per student/TMA example
# ============================================================
def build_dataset(raw_dir: Path, output_path: Path) -> int:
    # ASSESSMENT METADATA: keep the FFF TMAs and order them by deadline.
    by_presentation = defaultdict(list)
    assessments = {}
    for row in read_csv(raw_dir / "assessments.csv"):
        if row["code_module"] != MODULE or row["code_presentation"] not in PRESENTATIONS:
            continue
        if row["assessment_type"] != "TMA" or row["date"] == "?":
            continue
        row["date"] = int(row["date"])
        row["weight"] = float(row["weight"])
        assessments[row["id_assessment"]] = row
        by_presentation[row["code_presentation"]].append(row)

    for rows in by_presentation.values():
        rows.sort(key=lambda item: (item["date"], item["id_assessment"]))

    # SUBMISSION HISTORY: collect target scores and earlier submitted scores.
    # Earlier-only history prevents future assessment results leaking into features.
    target_scores = {}
    histories = defaultdict(list)
    for row in read_csv(raw_dir / "studentAssessment.csv"):
        assessment = assessments.get(row["id_assessment"])
        if assessment is None or row["score"] == "?":
            continue
        score = float(row["score"])
        key = (assessment["code_presentation"], row["id_student"])
        target_scores[(key[0], key[1], row["id_assessment"])] = score

        # A score is usable as a feature only if it was submitted before the
        # target deadline. Exclude banked records from historical features
        # because their original submission timing is not represented here.
        if row["date_submitted"] != "?" and row.get("is_banked", "0") == "0":
            submitted_day = int(row["date_submitted"])
            histories[key].append(
                (assessment["date"], submitted_day, row["id_assessment"], score)
            )

    for values in histories.values():
        values.sort(key=lambda item: (item[1], item[0], item[2]))

    # VLE ACTIVITY FEATURES: count clicks only in windows before each deadline.
    # Aggregate VLE clicks in [deadline - 28, deadline), excluding deadline day.
    activity = defaultdict(lambda: [0, 0])
    for row in read_csv(raw_dir / "studentVle.csv"):
        presentation = row["code_presentation"]
        if row["code_module"] != MODULE or presentation not in by_presentation or row["date"] == "?":
            continue
        day = int(row["date"])
        student = row["id_student"]
        clicks = int(row["sum_click"])
        for assessment in by_presentation[presentation]:
            deadline = assessment["date"]
            if deadline - WINDOW_DAYS <= day < deadline:
                bucket = activity[(presentation, student, assessment["id_assessment"])]
                bucket[0] += clicks
                if day >= deadline - RECENT_DAYS:
                    bucket[1] += clicks

    # OUTPUT SCHEMA + EXPORT: write one feature row for each scored student/TMA.
    fields = [
        "presentation", "split", "student_id", "assessment_id",
        "assessment_number", "assessment_day", "assessment_weight",
        "previous_score_count", "previous_score_mean", "previous_score_last",
        "clicks_preceding_28d", "clicks_preceding_7d", "target_score",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for presentation in PRESENTATIONS:
            for (pres, student), history in sorted(histories.items()):
                if pres != presentation:
                    continue
                for number, assessment in enumerate(by_presentation[presentation], start=1):
                    deadline = assessment["date"]
                    target_key = (presentation, student, assessment["id_assessment"])
                    if target_key not in target_scores:
                        continue
                    previous = [
                        score
                        for assessment_day, submitted_day, assessment_id, score in history
                        if assessment_id != assessment["id_assessment"]
                        and assessment_day < deadline
                        and submitted_day < deadline
                    ]
                    all_clicks, recent_clicks = activity[target_key]
                    writer.writerow({
                        "presentation": presentation,
                        "split": "train" if presentation == TRAIN_PRESENTATION else "test",
                        "student_id": student,
                        "assessment_id": assessment["id_assessment"],
                        "assessment_number": number,
                        "assessment_day": deadline,
                        "assessment_weight": assessment["weight"],
                        "previous_score_count": len(previous),
                        "previous_score_mean": round(sum(previous) / len(previous), 4) if previous else "",
                        "previous_score_last": previous[-1] if previous else "",
                        "clicks_preceding_28d": all_clicks,
                        "clicks_preceding_7d": recent_clicks,
                        "target_score": target_scores[target_key],
                    })
                    count += 1
    return count


# COMMAND-LINE ENTRY POINT: parse paths and build the processed dataset.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/fff_tma_examples.csv"))
    args = parser.parse_args()
    count = build_dataset(args.raw_dir, args.output)
    print(f"Wrote {count:,} examples to {args.output}")


if __name__ == "__main__":
    main()
