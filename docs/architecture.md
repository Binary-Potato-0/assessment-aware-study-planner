# Architecture and explanation guide

## End-to-end flow

1. **Data preparation:** read OULAD assessment, submission and VLE tables.
   Build one example per student/TMA with features available before the target
   deadline. Historical marks are included only when the assessment was due
   earlier and the student submitted it before the current deadline.
2. **Neural Network:** an MLP maps eight numeric features to an estimated mark.
   Training uses 2013J; evaluation uses the later 2014J presentation.
3. **Search:** a state is a tuple of session counts, e.g. `(2, 1, 3)`.
   A successor adds one session to one assignment. UCS prioritizes total
   sessions used; Greedy prioritizes an estimate of remaining sessions.
4. **CSP:** each proposed session becomes a variable. Its domain is available
   weekly slots before the deadline. Constraints enforce unique slots and
   chronological order for sessions belonging to the same assignment.
5. **Integration:** Search checks whether a state reaches the mark target.
   If so, the CSP attempts to schedule it. If no schedule exists, the goal is
   rejected and Search continues.

## Neural Network training concepts

- **Features:** numeric inputs describing the student and assessment.
- **Target:** the observed assessment score.
- **Forward pass:** compute the network's predicted score.
- **Loss:** quantify the difference between prediction and actual score.
- **Backpropagation:** calculate how network weights contributed to the error.
- **Optimizer:** update weights to reduce the loss over training iterations.
- **Evaluation:** MAE and RMSE measure prediction error; R² compares model
  performance to predicting the test-set mean.

## Search concepts

- A **state** represents a candidate allocation.
- An **action** adds one session.
- **Path cost g(n)** is the total number of sessions allocated.
- **Heuristic h(n)** estimates remaining sessions from mark deficits and
  explicit per-session gain assumptions.
- UCS is optimal here because every action has equal cost and the state space
  is finite. Greedy is guided by h(n), but is not guaranteed optimal.

## CSP concepts

- **Variable:** one study session.
- **Domain:** candidate weekly time slots before its assignment deadline.
- **Constraint:** no two sessions share a slot; sessions for one assignment
  must occur in chronological order.
- **Backtracking:** assign a value, recurse, and undo it if it leads to failure.
- **Forward Checking:** after an assignment, remove incompatible values from
  neighboring unassigned variables.
- **AC-3:** repeatedly remove values that have no supporting value in a
  neighboring variable's domain until domains are consistent or one is empty.

## Critical limitation

OULAD is observational. VLE clicks and marks do not reveal the causal effect
of hypothetical extra study sessions. The MLP predicts observed marks.
The per-session gains used by the planning scenario are assumptions, not
causal estimates learned from the dataset.
