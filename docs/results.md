# Results Snapshot

This snapshot summarizes one local run of the configured FAERS signal-ranking pipeline. It is included so reviewers can quickly see the project output shape. It is not clinical validation.

## Benchmark Summary

The evaluation compares ranked candidate signals against curated known-positive and weak-control GLP-1 drug-event pairs.

| Metric | Result |
|:---|:---|
| Known-positive pairs in benchmark | 6 |
| Known positives recovered in top 10 | 2 / 6 |
| Known positives recovered in top 20 | 5 / 6 |
| Weak controls in top 5 | 0 |

## Known-Positive Recovery

| Drug | Event family | Rank | Top 10 | Top 20 |
|:---|:---|---:|:---:|:---:|
| semaglutide | pancreatitis | 13 | No | Yes |
| liraglutide | gallbladder disease | 18 | No | Yes |
| semaglutide | gallbladder disease | 8 | Yes | Yes |
| liraglutide | pancreatitis | 17 | No | Yes |
| exenatide | pancreatitis | Not ranked | No | No |
| dulaglutide | pancreatitis | 4 | Yes | Yes |

## Example Top-Ranked Signals

| Rank | Drug | Event family | FAERS cases | ROR | ROR lower CI | Seriousness rate |
|---:|:---|:---|---:|---:|---:|---:|
| 1 | dulaglutide | ileus | 100 | 72.868 | 58.197 | 100.00% |
| 2 | dulaglutide | severe nausea/vomiting | 747 | 20.119 | 18.091 | 95.31% |
| 3 | dulaglutide | intestinal obstruction | 134 | 28.421 | 23.614 | 100.00% |
| 4 | dulaglutide | pancreatitis | 219 | 11.828 | 10.217 | 95.89% |
| 5 | dulaglutide | dehydration | 177 | 11.097 | 9.456 | 98.31% |
| 6 | semaglutide | intestinal obstruction | 125 | 10.070 | 8.363 | 100.00% |
| 7 | semaglutide | ileus | 53 | 13.338 | 10.019 | 100.00% |
| 8 | semaglutide | gallbladder disease | 117 | 8.780 | 7.255 | 99.15% |

## Interpretation

The benchmark suggests the ranking pipeline recovers most known-positive examples within the top 20 while keeping weak controls out of the strongest ranks. Misses and unexpected rankings are expected in a spontaneous-reporting dataset and require manual review for mapping quality, quarter-specific reporting effects, and reporting bias.
