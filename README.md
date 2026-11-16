# Assessment-Aware Study Planner

A decision-support prototype that combines a Neural Network (NN), Search and
Constraint Satisfaction (CSP) to explore study-session allocations and build a
weekly timetable for upcoming assignments.

## Current scope

- OULAD dataset: FFF 2013J for training and FFF 2014J as a held-out presentation.
- MLP regression model predicts observed TMA marks from prior submitted marks
  and pre-deadline VLE activity.
- Uniform-Cost Search (UCS) and Greedy Best-First Search explore session counts.
- CSP scheduling uses Backtracking, Forward Checking and AC-3.
- Tests cover search, scheduling, integration, and impossible cases.

## Setup (Windows CMD)

Place the original OULAD CSV files in `data/raw/`. Keep the original files
unchanged. Install dependencies:

```cmd
python -m pip install -r requirements.txt
python data\prepare_dataset.py
python nn\train.py
python run_demo.py
python -m unittest discover -s tests -v
```

The training command writes evaluation metrics to `nn/results.json` and the
trained pipeline to `nn/model.joblib`. The integrated demo loads the trained
model, predicts baseline marks for one held-out student, then runs Search and
CSP. The demo's per-session gains are explicit scenario assumptions.

## Components

- `data/prepare_dataset.py`: creates leakage-aware examples from raw OULAD.
- `nn/train.py`: trains and evaluates the MLP against a student-history baseline.
- `nn/predict.py`: loads the saved MLP and predicts marks for a held-out student.
- `search/core.py`: UCS and Greedy Best-First allocation search.
- `csp/solver.py`: timetable CSP with Backtracking, Forward Checking and AC-3.
- `integration/planner.py`: rejects unschedulable goal states and continues Search.
- `run_demo.py`: integrated entry point.

## Scientific limitation

OULAD records observational marks and online-learning clicks. It does not
contain randomized interventions or direct measurements of additional study
sessions. Therefore the MLP predicts observed marks; it does not estimate the
causal effect of adding a study session. The session-to-mark gains in the
planning demonstration are transparent scenario assumptions and must not be
reported as learned causal effects.

The prototype is decision support, not a guarantee of student performance.
