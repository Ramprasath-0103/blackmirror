"""
The building blocks of the BlackMirror experiment.

Each module does one job and maps to one section of the plan:

    dataset.py      Phase 1.1-1.3  load, clean, subsample and split Adult
    generation.py   Phase 1.4-1.5  SDV metadata and the recursive synthetic chain
    mixing.py       Phase 1.6      build a real/synthetic training mixture
    encoding.py     Phase 2.1      turn text categories into numbers
    model.py        Phase 2.2/2.5  train a Random Forest and score it
    search.py       Phase 2.2-2.4  search space, objective, Hill Climbing, Sim. Annealing
    collapse.py     Phase 3.1      the Collapse Index
    forecasting.py  Phase 3.2      the LSTM forecast and the naive baseline
    plots.py        Phase 3.3      the two figures
    report.py       Phase 3.3      the summary table and written findings

The three scripts in the project folder (prepare_data.py, run_experiment.py,
analyze.py) import these modules and run them in the right order.
"""
