# BlackMirror research plots

This folder creates five standalone, high-resolution figures from
`results/results.csv`. It does not change the experiment results or overwrite
the two standard analysis figures in `results/plots/`.

## Generate the figures

Run these commands from the project root after completing the experiment:

```bash
python -m plotting.generate_plots
```

The PNG files are written to `plotting/plots/`. The script checks for all 62
experiment runs before generating figures and reports a clear error if results
are missing or incomplete.

## Figures

1. **`01_collapse_index_by_generation.png`** — Recreates the README's
   generation-versus-Collapse-Index figure. Color indicates the real/synthetic
   ratio; solid and dashed lines distinguish Hill Climbing (HC) and Simulated
   Annealing (SA). A value above zero means lower test F1 than the matching
   algorithm's real-only baseline.
2. **`02_hc_vs_sa.png`** — Recreates the README's algorithm comparison. Each
   point is a matched training run: the left panel compares best validation
   F1, and the right panel compares evaluations used. Above the diagonal means
   SA has a higher value: a higher score in the left panel, but more search
   effort in the right. The note reports SA's validation-F1 wins, ties, and
   losses across the 30 synthetic-data runs.
3. **`03_mean_degradation_by_ratio.png`** — Summarizes mean Collapse Index
   across G1–G10 as synthetic share increases. Vertical ranges show the minimum
   and maximum across generations; they are **not** confidence intervals or
   repeated-run uncertainty.
4. **`04_test_f1_by_generation.png`** — Shows absolute test F1 for each
   generation and ratio, split into one panel per search algorithm. The dotted
   line is the same algorithm's 100%-real baseline.
5. **`05_paired_test_f1_difference.png`** — Compares SA and HC on the same
   generation and ratio using `SA test F1 − HC test F1`. Positive values favor
   SA; negative values favor HC. Each point comes from one run.

## Interpretation note

The experiment currently has one run for each generation, ratio, and
algorithm. These figures describe the observed runs; they do not show
confidence intervals across repeated experiments. In particular, the range in
figure 3 describes variation between generations, not statistical uncertainty.
For inferential comparisons, repeat the experiment using multiple controlled
random seeds and plot uncertainty across those repeated runs.
