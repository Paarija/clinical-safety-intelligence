"""Regression coverage for preferred-term event matching in retrievers."""

from __future__ import annotations

from clinical_safety.evidence.retrievers.fda_retriever import FDARetriever
from clinical_safety.evidence.retrievers.pubmed_retriever import PubMedRetriever


def test_pubmed_build_query_accepts_preferred_terms_as_or_set() -> None:
    retriever = PubMedRetriever()
    query = retriever._build_query("semaglutide", ["vomiting", "nausea"])

    assert '"semaglutide"[tiab]' in query
    assert '"semaglutide"[MeSH Terms]' in query
    assert '"vomiting"[tiab]' in query
    assert '"vomiting"[MeSH Terms]' in query
    assert '"nausea"[tiab]' in query
    assert '"nausea"[MeSH Terms]' in query
    assert query.count(" OR ") >= 3


def test_fda_event_match_accepts_any_preferred_term_from_list() -> None:
    retriever = FDARetriever()
    label = {
        "warnings_and_precautions": [
            "Mild headache and rash were observed.",
        ],
        "adverse_reactions": [
            "Subjects reported severe bowel obstruction requiring intervention.",
        ],
    }

    section, snippet = retriever._find_event_in_label(label, ["nausea", "bowel obstruction"])

    assert section == "adverse_reactions"
    assert snippet
    assert "bowel obstruction" in snippet
