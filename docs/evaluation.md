# Evaluation

The evaluation module provides an engineering benchmark for the signal-ranking pipeline. It compares ranked candidate signals against a small curated set of known-positive and weak-control GLP-1 drug-event pairs.

Run it after FAERS acquisition and signal ranking:

```bash
python3 -m clinical_safety.evaluation.runner
```

## Benchmark Sets

Known-positive examples are drug-event pairs with label, regulatory, or literature support. Weak controls are pairs that should not rank highly unless the data or mapping logic surfaces a plausible reason.

| Set | Purpose |
|:---|:---|
| Known positives | Check whether expected GLP-1 safety associations appear in the ranked list |
| Weak controls | Check whether unrelated pairs are kept out of the highest ranks |

## Metrics

| Metric | Meaning |
|:---|:---|
| Top-K recovery | Number of known-positive pairs recovered in the top K ranked signals |
| Weak controls in top 5 | Count of weak-control pairs that appear among the strongest signals |
| Recovery details | Rank-by-rank detail for each curated known-positive pair |

The output is written to:

```text
data/processed/analytics/evaluation_results.json
```

## Interpretation

These metrics are useful for regression checks and project demonstration, but they do not prove clinical validity. Missed known positives may reflect quarter-specific reporting, low case counts, mapping gaps, or threshold choices. Highly ranked unexpected pairs require manual review rather than automatic dismissal or acceptance.
