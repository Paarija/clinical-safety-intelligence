from __future__ import annotations

from typing import Protocol

from clinical_safety.common.types import EvidenceDocument


class EvidenceRetriever(Protocol):
    """External evidence source with the retriever call contract used by the graph."""

    def retrieve(
        self,
        drug_id: str,
        event_id: str,
        drug_name: str,
        event_name: str,
    ) -> list[EvidenceDocument]:
        """Return evidence documents for one drug-event pair."""


__all__ = ["EvidenceRetriever"]
