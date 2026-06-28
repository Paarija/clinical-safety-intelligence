from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class ClinicalTrialsIdentificationModule(BaseModel):
    """ClinicalTrials.gov identification payload with the required NCT id."""

    model_config = ConfigDict(extra="allow")

    nctId: str


class ClinicalTrialsProtocolSection(BaseModel):
    """ClinicalTrials.gov protocol section fields used downstream."""

    model_config = ConfigDict(extra="allow")

    identificationModule: ClinicalTrialsIdentificationModule


class ClinicalTrialsStudy(BaseModel):
    """Minimal ClinicalTrials.gov study shape required by the pipeline."""

    model_config = ConfigDict(extra="allow")

    protocolSection: ClinicalTrialsProtocolSection
    resultsSection: dict[str, Any] | None = None

    @property
    def nct_id(self) -> str:
        return self.protocolSection.identificationModule.nctId


class ClinicalTrialsResponseEnvelope(BaseModel):
    """Lightweight ClinicalTrials.gov API response envelope."""

    model_config = ConfigDict(extra="allow")

    studies: list[ClinicalTrialsStudy]
    nextPageToken: str | None = None
    totalCount: int | None = None


class ClinicalTrialsRawFileEnvelope(BaseModel):
    """Lightweight raw ClinicalTrials JSON file envelope."""

    model_config = ConfigDict(extra="allow")

    studies: list[dict[str, Any]]
    drug_id: str | None = None


def format_validation_error(exc: ValidationError) -> str:
    """Format Pydantic validation details for concise data-source errors."""
    details: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ())) or "<response>"
        details.append(f"{loc}: {err.get('msg', 'invalid value')}")
    return "; ".join(details)
