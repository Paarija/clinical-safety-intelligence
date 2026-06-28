# Limitations

This project is designed for safety-signal triage and analyst review. It should not be used for clinical decisions, prescribing guidance, regulatory submissions, or claims of causality.

## FAERS Does Not Establish Causality

FAERS/AEMS is a spontaneous adverse-event reporting system. Reports describe suspected associations, not confirmed drug-event causation. Case counts, ROR, and PRR measure disproportionate reporting, not incidence or patient-level risk.

## Reporting Bias Is Expected

Signals can be distorted by market share, media attention, litigation, existing label warnings, manufacturer reporting patterns, and differences between consumer and clinician reports. The pipeline can surface these signals, but it cannot fully correct those biases.

## Mapping Is Approximate

FAERS drug names and reaction terms are messy free text. The project normalizes them with aliases, fuzzy matching, and event-family mappings, but ambiguous or low-confidence mappings can still occur.

## Trial Comparison Is Incomplete

ClinicalTrials.gov enrichment depends on posted results and adverse-event tables. Trial populations, follow-up windows, and event definitions often differ from post-market populations.

## LLM Synthesis Requires Review

Gemini synthesis is constrained to retrieved evidence and guardrails, but LLM output can still omit nuance or misstate source material. Generated summaries should be reviewed against the cited source documents.

## Evaluation Is Engineering-Oriented

The included benchmark checks whether known-positive and weak-control pairs behave plausibly in the ranking pipeline. It is not a clinical validation package.
