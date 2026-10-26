
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from src.model import f1, train_random_forest

State = tuple[int, int, int]
PARAM_NAMES = list(config.GRID) 

def state_to_params(state: State) -> dict:

    return {name: config.GRID[name][index] for name, index in zip(PARAM_NAMES, state)}


def neighbors(state: State) -> list[State]:

    result = []
    for position, name in enumerate(PARAM_NAMES):
        for step in (-1, +1):
            new_index = state[position] + step
            if 0 <= new_index < len(config.GRID[name]):
                neighbor = list(state)
                neighbor[position] = new_index
                result.append(tuple(neighbor))
    return result


class Objective:

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

def hill_climbing(objective: Objective, log: list | None = None) -> SearchResult:

    current = config.START_STATE
    f_current = objective(current)

    while True:
        scores = {}
        for neighbor in neighbors(current):
            if objective.can_score(neighbor):
                scores[neighbor] = objective(neighbor)

        if not scores:
            break 
        best_neighbor = max(scores, key=scores.get)
        moved = scores[best_neighbor] > f_current
        if log is not None:
            log.append({"state": current, "f1": f_current, "neighbors": scores,
                        "best": best_neighbor, "moved": moved, "evals_used": objective.evals_used})
        if moved:
            current, f_current = best_neighbor, scores[best_neighbor]
        else:
            break

    return SearchResult(current, f_current, objective.evals_used)

def simulated_annealing(objective: Objective, log: list | None = None) -> SearchResult:

    rng = np.random.default_rng(config.SEED)
    current = config.START_STATE
    f_current = objective(current)
    best, f_best = current, f_current
    temperature = config.SA_START_TEMPERATURE

    for step in range(1, config.SA_MAX_STEPS + 1):
        if not objective.budget_left:
            break

        options = neighbors(current)
        candidate = options[rng.integers(len(options))]
        f_candidate = objective(candidate)
        delta = f_candidate - f_current

        if delta > 0:
            accepted, probability, roll = True, 1.0, None
        else:
            probability = math.exp(delta / temperature)
            roll = rng.random()
            accepted = roll < probability

        if accepted:
            current, f_current = candidate, f_candidate
        new_best = f_current > f_best
        if new_best:
            best, f_best = current, f_current

        if log is not None:
            log.append({"step": step, "temperature": temperature, "candidate": candidate,
                        "f1": f_candidate, "delta": delta, "probability": probability, "roll": roll,
                        "accepted": accepted, "new_best": new_best, "evals_used": objective.evals_used})

        temperature *= config.SA_COOLING_RATE

    return SearchResult(best, f_best, objective.evals_used)


ALGORITHMS = {"HC": hill_climbing, "SA": simulated_annealing}
