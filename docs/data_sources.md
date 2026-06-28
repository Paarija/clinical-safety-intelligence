# Data Sources

The pipeline uses public data sources. Source paths, URLs, and request settings are configured in `configs/data_sources.yaml`.

## FAERS/AEMS

FDA FAERS/AEMS quarterly public data is the primary post-market source.

- Public download: <https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html>
- Format: ZIP archive with ASCII `$`-delimited files
- Used for: report deduplication, drug/event normalization, contingency tables, disproportionality metrics, seriousness summaries

Main files:

| File family | Purpose |
|:---|:---|
| `DEMO` | Case metadata and report dates |
| `DRUG` | Drug names and role codes |
| `REAC` | Reaction preferred terms |
| `OUTC` | Outcome codes such as death or hospitalization |
| `INDI`, `THER`, `RPSR` | Optional indication, therapy, and report-source context |

Important caveat: FAERS is a spontaneous reporting system. It supports signal detection, not causality or incidence estimation.

## ClinicalTrials.gov

ClinicalTrials.gov is used as optional enrichment for trial adverse-event results.

- API: <https://clinicaltrials.gov/data-api>
- Used for: trial arms, adverse-event rates, comparator context, posted results availability

Important caveat: not all trials post results, and adverse-event terminology can differ from FAERS terms.

## FDA Labels and Safety Communications

The project retrieves regulatory context from openFDA labels and FDA safety communication pages.

- openFDA drug labels: <https://open.fda.gov/apis/drug/label/>
- FDA safety communications: <https://www.fda.gov/drugs/drug-safety-and-availability/drug-safety-communications>
- Used for: warnings, precautions, adverse-reaction text, and regulatory context

Important caveat: labels and communications are not a complete substitute for formal regulatory review.

## PubMed

PubMed / NCBI E-utilities are used for peer-reviewed literature retrieval.

- API overview: <https://www.ncbi.nlm.nih.gov/home/develop/api/>
- Used for: abstracts and literature context around drug-event pairs
- Optional key: `NCBI_API_KEY` increases request allowance

Important caveat: abstracts may be incomplete, and retrieved literature still needs manual relevance review.
