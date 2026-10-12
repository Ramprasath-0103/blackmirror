"""
search.py - find good Random Forest hyperparameters (plan Phase 2.2 - 2.4).

The search is framed as a classic AI search problem:

    State          a triple of indices into config.GRID, e.g. (1, 1, 0)
    Initial state  config.START_STATE -> (100 trees, depth 4, split 2)
    Action         move ONE hyperparameter ONE step up or down
    Evaluation     validation F1 of a forest trained with those settings
    Goal           the highest validation F1
    Termination    40 unique evaluations, or (Hill Climbing only) no better neighbour

Contents:
    state_to_params(state)        (1, 1, 0) -> {"n_estimators": 100, ...}
    neighbors(state)              every state one step away
    Objective                     scores states, remembers scores, enforces the budget
    hill_climbing(objective)      always move to the best neighbour, stop at a peak
    simulated_annealing(...)      random moves, sometimes accepts worse ones early on
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from src.model import f1, train_random_forest

State = tuple[int, int, int]
PARAM_NAMES = list(config.GRID)  # ["n_estimators", "max_depth", "min_samples_split"]


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
def state_to_params(state: State) -> dict:
    """Translate index triple -> real hyperparameter values."""
    return {name: config.GRID[name][index] for name, index in zip(PARAM_NAMES, state)}


def neighbors(state: State) -> list[State]:
    """
    All states that differ by +1 or -1 in exactly one index (at most 6).

    Example: neighbors((0, 1, 0)) -> [(1, 1, 0), (0, 0, 0), (0, 2, 0), (0, 1, 1)]
    (no (-1, 1, 0) because index -1 would fall outside the grid).
    """
    result = []
    for position, name in enumerate(PARAM_NAMES):
        for step in (-1, +1):
            new_index = state[position] + step
            if 0 <= new_index < len(config.GRID[name]):
                neighbor = list(state)
                neighbor[position] = new_index
                result.append(tuple(neighbor))
    return result


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
class Objective:
    """
    Turns a state into a score: train a forest on the mixture, return F1 on V.

    It also keeps a cache (state -> F1). Hill Climbing looks at the same
    neighbours again and again; thanks to the cache a repeated state is free
    and only UNIQUE evaluations count against the budget. That keeps the
    comparison between the two algorithms fair.

    Use it like a function:   score = objective(state)
    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series, budget: int = config.EVAL_BUDGET):
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.budget = budget
        self.cache: dict[State, float] = {}

    @property
    def evals_used(self) -> int:
        return len(self.cache)

    @property
    def budget_left(self) -> bool:
        return self.evals_used < self.budget

    def can_score(self, state: State) -> bool:
        """True if scoring this state is allowed (already cached, or budget left)."""
        return state in self.cache or self.budget_left

    def __call__(self, state: State) -> float:
        if state not in self.cache:
            if not self.budget_left:
                raise RuntimeError("Evaluation budget is used up")
            model = train_random_forest(state_to_params(state), self.X_train, self.y_train)
            self.cache[state] = f1(model, self.X_val, self.y_val)
        return self.cache[state]


@dataclass
class SearchResult:
    best_state: State
    best_val_f1: float
    evals_used: int

    @property
    def best_params(self) -> dict:
        return state_to_params(self.best_state)


# ---------------------------------------------------------------------------
# Hill Climbing (steepest ascent)
# ---------------------------------------------------------------------------
def hill_climbing(objective: Objective) -> SearchResult:
    """
    Score every neighbour, move to the best one if it beats the current state.
    Stop at a local optimum (no better neighbour) or when the budget runs out.
    """
    current = config.START_STATE
    f_current = objective(current)

    while True:
        scores = {}
        for neighbor in neighbors(current):
            if objective.can_score(neighbor):
                scores[neighbor] = objective(neighbor)

        if not scores:
            break  # budget used up, nothing left to look at

        best_neighbor = max(scores, key=scores.get)
        if scores[best_neighbor] > f_current:
            current, f_current = best_neighbor, scores[best_neighbor]
        else:
            break  # local optimum: no neighbour is better

    return SearchResult(current, f_current, objective.evals_used)


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------
def simulated_annealing(objective: Objective) -> SearchResult:
    """
    Pick a random neighbour each step.
      - better?  always move there
      - worse?   move anyway with probability exp(delta / T)

    T starts at 0.05 (about the size of typical F1 differences) and shrinks
    by 10% every step: early on SA explores, later it only climbs.
    Because SA can walk downhill, it returns the BEST state it ever saw,
    not the state where it ended.

    Same start, neighbours, budget and objective as Hill Climbing - only the
    rule for accepting a move is different.
    """
    rng = np.random.default_rng(config.SEED)
    current = config.START_STATE
    f_current = objective(current)
    best, f_best = current, f_current
    temperature = config.SA_START_TEMPERATURE

    for _ in range(config.SA_MAX_STEPS):
        if not objective.budget_left:
            break

        options = neighbors(current)
        candidate = options[rng.integers(len(options))]
        f_candidate = objective(candidate)
        delta = f_candidate - f_current

        if delta > 0 or rng.random() < math.exp(delta / temperature):
            current, f_current = candidate, f_candidate
        if f_current > f_best:
            best, f_best = current, f_current

        temperature *= config.SA_COOLING_RATE

    return SearchResult(best, f_best, objective.evals_used)


ALGORITHMS = {"HC": hill_climbing, "SA": simulated_annealing}
