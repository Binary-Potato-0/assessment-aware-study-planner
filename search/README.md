# Search component

## What Search is responsible for

The neural network estimates a mark. Search chooses a low-cost vector of
additional study sessions, one count per assignment, whose predicted marks
meet the requested target.

A state is a tuple such as `(2, 1, 3)`: two sessions for assignment 1, one
for assignment 2, and three for assignment 3. The initial state is all zeros.
An action increments exactly one assignment's session count by one.

## Algorithms

- **Uniform-Cost Search (UCS):** priority is `g(n)`, the total sessions
  already allocated. Each action costs one, so the first goal popped has the
  minimum possible total number of sessions.
- **Greedy Best-First Search:** priority is `h(n)`, an estimate of remaining
  sessions. It sums each assignment's mark deficit divided by an assumed
  gain per session. It may find a goal quickly, but is not guaranteed optimal.

Both algorithms report nodes generated and expanded. The search space is
finite because each assignment has a configurable session limit.

## Predictor boundary and scientific limitation

Search accepts a predictor function instead of depending directly on the
neural-network code. This keeps the algorithm testable and lets the integration
layer connect components later.

**Important:** OULAD contains observed marks and VLE clicks, not randomized
additional study sessions. A model trained on those observations cannot
identify the causal mark increase from one hypothetical extra session.
The demo therefore uses a clearly labelled illustrative predictor. Its
per-session gains are scenario assumptions, not learned causal effects.
The integrated planner must preserve this distinction and document any
session-to-activity assumption.

## Run

```cmd
python -m search.demo
python -m unittest discover -s tests -p "test_search.py" -v
```
