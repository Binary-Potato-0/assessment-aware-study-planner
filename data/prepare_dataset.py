"""Build leakage-aware student/assessment examples from the local OULAD CSV files.

This first pipeline intentionally uses only observed pre-deadline information.
It does NOT claim that clicks or hypothetical study sessions cause higher marks.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

MODULE = "FFF"
PRESENTATIONS = ("2013J", "2014J")
TRAIN_PRESENTATION = "2013J"
TEST_PRESENTATION = "2014J"
WINDOW_DAYS = 28
RECENT_DAYS = 7


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def build_dataset(raw_dir: Path, output_path: Path) -> int:
    assessments = {}
    by_presentation = defaultdict(list)
    for row in read_csv(raw_dir / "assessments.csv"):
        if (row["code_module"], row["code_presentation"]) != (MODULE, TRAIN_PRESENTATION) and (
            row["code_module"], row["code_presentation"]
        ) != (MODULE, TEST_PRESENTATION):
            continue
        if row["assessment_type"] != "TMA" or row["date"] == "?":
            continue
        row["date"] = int(row["date"])
        row["weight"] = float(row["weight"])
        assessments[row["id_assessment"]] = row
        by_presentation[row["code_presentation"]].append(row)

    for rows in by_presentation.values():
        rows.sort(key=lambda item: (item["date"], item["id_assessment"]))

    scores = defaultdict(list)
    for row in read_csv(raw_dir / "studentAssessment.csv"):
        assessment = assessments.get(row["id_assessment"])
        if assessment is None or row["score"] == "?":
            continue
        scores[(assessment["code_presentation"], row["id_student"])].append(
            (assessment["date"], float(row["score"]))
        )
    for values in scores.values():
        values.sort()

    site_types = {}
    for row in read_csv(raw_dir / "vle.csv"):
        if row["code_module"] == MODULE and row["code_presentation"] in PRESENTATIONS:
            site_types[(row["code_presentation"], row["id_site"])] = row["activity_type"]

    # (presentation, student, assessment) -> [all-window clicks, recent clicks]
    activity = defaultdict(lambda: [0, 0])
    target_assessments = defaultdict(list)
    for presentation, rows in by_presentation.items():
        for assessment in rows:
            target_assessments[presentation].append(assessment)

    for row in read_csv(raw_dir / "studentVle.csv"):
        if row["code_module"] != MODULE or row["code_presentation"] not in PRESENTATIONS:
            continue
        if row["date"] == "?":
            continue
        day = int(row["date"])
        presentation = row["code_presentation"]
        student = row["id_student"]
        for assessment in target_assessments[presentation]:
            deadline = assessment["date"]
            if deadline - WINDOW_DAYS <= day < deadline:
                key = (presentation, student, assessment["id_assessment"])
                clicks = int(row["sum_click"])
                activity[key][0] += clicks
                if day >= deadline - RECENT_DAYS:
                    activity[key][1] += clicks

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
            for student in sorted({student for (pres, student) in scores if pres == presentation}):
                history = scores[(presentation, student)]
                for number, assessment in enumerate(by_presentation[presentation], start=1):
                    target = next(
                        (score for day, score in history if day == assessment["date"]
                         and False), None
                    )
                    # Match the target by assessment id, not merely by deadline.
                    target_row = None
                    for row in read_csv(raw_dir / "studentAssessment.csv"):
                        if row["id_assessment"] == assessment["id_assessment"] and row["id_student"] == student:
                            if row["score"] != "?":
                                target_row = float(row["score"])
                            break
                    if target_row is None:
                        continue
                    previous = [score for day, score in history if day < assessment["date"]]
                    all_clicks, recent_clicks = activity[
                        (presentation, student, assessment["id_assessment"])
                    ]
                    writer.writerow({
                        "presentation": presentation,
                        "split": "train" if presentation == TRAIN_PRESENTATION else "test",
                        "student_id": student,
                        "assessment_id": assessment["id_assessment"],
                        "assessment_number": number,
                        "assessment_day": assessment["date"],
                        "assessment_weight": assessment["weight"],
                        "previous_score_count": len(previous),
                        "previous_score_mean": round(sum(previous) / len(previous), 4) if previous else "",
                        "previous_score_last": previous[-1] if previous else "",
                        "clicks_preceding_28d": all_clicks,
                        "clicks_preceding_7d": recent_clicks,
                        "target_score": target_row,
                    })
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/fff_tma_examples.csv"))
    args = parser.parse_args()
    count = build_dataset(args.raw_dir, args.output)
    print(f"Wrote {count:,} examples to {args.output}")


if __name__ == "__main__":
    main()
