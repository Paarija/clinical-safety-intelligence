"""
evidence/retrievers/fda_retriever.py

Retrieves FDA regulatory evidence for a drug + adverse event signal.

Two retrieval strategies:
  1. openFDA drug label API (/drug/label.json)
     - Searches warnings_and_precautions, boxed_warnings, adverse_reactions
       sections for mentions of the drug and event term.
     - Public API, no key required at low query volume.

  2. FDA Drug Safety Communications page (HTML scrape)
     - Searches for drug name in safety communication titles.
     - Returns matched communications as EvidenceDocument objects.

Returns a combined list of EvidenceDocument objects.

Usage:
    from clinical_safety.evidence.retrievers.fda_retriever import FDARetriever
    retriever = FDARetriever()
    docs = retriever.retrieve(drug_id="semaglutide", event_id="pancreatitis",
                              drug_name="semaglutide", event_name="pancreatitis")
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from clinical_safety.common.config import get_config
from clinical_safety.common.exceptions import EvidenceRetrievalError
from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths
from clinical_safety.common.types import CredibilityClass, EvidenceDocument

logger = get_logger(__name__)

_CACHE_TTL = timedelta(days=30)
_CACHE_NOTE = "[Retrieved from FDA cache after API failure; verify source freshness.]"

# openFDA label sections to search for event mentions
_LABEL_SECTIONS = [
    "warnings_and_precautions",
    "boxed_warnings",
    "adverse_reactions",
    "warnings",
    "precautions",
]


class FDARetriever:
    """
    Retrieves FDA label and safety communication evidence via public APIs.
    """

    def __init__(self, paths: Paths | None = None) -> None:
        cfg = get_config()
        self._cfg = cfg.data_sources.fda
        self._delay = self._cfg.request_delay_sec
        self._cache_dir = (paths or Paths()).external / "cache" / "fda"


    @staticmethod
    def _normalize_retrieval_terms(
        event_terms: list[str] | None,
        fallback_event_name: str | None,
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in event_terms or []:
            if not isinstance(raw, str):
                continue
            term = raw.replace('"', "").strip()
            if not term or term in seen:
                continue
            seen.add(term)
            normalized.append(term)

        if not normalized and fallback_event_name:
            fallback = fallback_event_name.replace('"', "").strip()
            if fallback and fallback not in seen:
                normalized.append(fallback)
        return normalized

    def retrieve(
        self,
        drug_id: str,
        event_id: str,
        drug_name: str,
        event_name: str,
        event_terms: list[str] | None = None,
    ) -> list[EvidenceDocument]:
        """
        Retrieve FDA label and safety communication evidence.

        API failures degrade per FDA sub-source: fresh cache is returned when
        available, otherwise the healthy sub-source can still contribute docs.
        """
        docs: list[EvidenceDocument] = []
        failures: list[str] = []
        normalized_event_terms = self._normalize_retrieval_terms(event_terms, event_name)

        try:
            label_docs = self._retrieve_label(
                drug_id,
                event_id,
                drug_name,
                normalized_event_terms,
            )
            self._write_cache("label", drug_id, event_id, label_docs)
        except EvidenceRetrievalError as exc:
            label_docs = self._read_cache("label", drug_id, event_id)
            if label_docs:
                logger.warning("openFDA label query failed; using cached FDA label docs: %s", exc)
            else:
                failures.append(str(exc))
        docs.extend(label_docs)

        time.sleep(self._delay)
        try:
            safety_docs = self._retrieve_safety_comms(drug_id, event_id, drug_name)
            self._write_cache("safety_comms", drug_id, event_id, safety_docs)
        except EvidenceRetrievalError as exc:
            safety_docs = self._read_cache("safety_comms", drug_id, event_id)
            if safety_docs:
                logger.warning("FDA safety comms query failed; using cached docs: %s", exc)
            else:
                failures.append(str(exc))
        docs.extend(safety_docs)

        if failures and not docs:
            raise EvidenceRetrievalError("; ".join(failures))
        if failures:
            logger.warning("FDA retrieval partially degraded: %s", "; ".join(failures))

        logger.info(
            "FDA retriever: %d label + %d safety comm docs for %s+%s",
            len(label_docs),
            len(safety_docs),
            drug_id,
            event_id,
        )
        return docs

    def _retrieve_label(
        self,
        drug_id: str,
        event_id: str,
        drug_name: str,
        event_terms: list[str],
    ) -> list[EvidenceDocument]:
        """
        Query the openFDA /drug/label.json endpoint.

        Searches for labels where openfda.generic_name matches the drug
        and one of the warning/AE sections mentions the event term.
        """
        search_query = f'openfda.generic_name:"{drug_name}"'
        params = {"search": search_query, "limit": 3}

        try:
            resp = requests.get(
                self._cfg.openfda_label_url,
                params=params,
                timeout=15,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "ClinicalSafetyIntelligence/0.1 (research pipeline)",
                },
            )
            if resp.status_code == 404:
                logger.debug("openFDA: no label found for '%s'", drug_name)
                return []

            resp.raise_for_status()
        except requests.RequestException as exc:
            raise EvidenceRetrievalError(f"openFDA label query failed: {exc}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise EvidenceRetrievalError(f"openFDA label response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Unexpected openFDA label response: expected JSON object")
        results = data.get("results")
        if not isinstance(results, list):
            raise ValueError("Unexpected openFDA label response: missing results list")
        docs: list[EvidenceDocument] = []

        for idx, label in enumerate(results):
            matched_section, snippet = self._find_event_in_label(label, event_terms)
            if not matched_section:
                continue

            set_id = label.get("set_id", f"label_{idx}")
            openfda = label.get("openfda", {})
            brand_names = openfda.get("brand_name", [drug_name])
            brand = brand_names[0] if brand_names else drug_name

            docs.append(
                EvidenceDocument(
                    doc_id=f"fda_label_{drug_id}_{set_id}",
                    source_type="fda_label",
                    title=f"FDA Prescribing Information: {brand} ({drug_name})",
                    url=(
                        "https://api.fda.gov/drug/label.json?search="
                        f'openfda.generic_name:"{drug_name}"&limit=1'
                    ),
                    identifier=set_id,
                    publication_date=None,
                    retrieved_date=datetime.now(UTC),
                    snippet=snippet,
                    credibility=CredibilityClass.FDA_LABEL,
                    drug_id=drug_id,
                    event_id=event_id,
                    accepted=True,
                )
            )
        return docs


    def _cache_path(self, source: str, drug_id: str, event_id: str):
        key = hashlib.sha256(f"{source}|{drug_id}|{event_id}".encode("utf-8")).hexdigest()
        return self._cache_dir / source / f"{key}.json"

    def _write_cache(
        self,
        source: str,
        drug_id: str,
        event_id: str,
        docs: list[EvidenceDocument],
    ) -> None:
        path = self._cache_path(source, drug_id, event_id)
        if not docs:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "cached_at": datetime.now(UTC).isoformat(),
                "source": source,
                "drug_id": drug_id,
                "event_id": event_id,
                "documents": [doc.model_dump(mode="json") for doc in docs],
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("FDA cache write failed for %s/%s/%s: %s", source, drug_id, event_id, exc)

    def _read_cache(
        self,
        source: str,
        drug_id: str,
        event_id: str,
    ) -> list[EvidenceDocument]:
        path = self._cache_path(source, drug_id, event_id)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(payload["cached_at"])
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=UTC)
            documents = payload["documents"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("FDA cache read failed for %s/%s/%s: %s", source, drug_id, event_id, exc)
            return []
        except ValueError as exc:
            logger.warning("FDA cache timestamp invalid for %s/%s/%s: %s", source, drug_id, event_id, exc)
            return []
        if datetime.now(UTC) - cached_at > _CACHE_TTL:
            logger.warning("FDA cache expired for %s/%s/%s", source, drug_id, event_id)
            return []
        if not isinstance(documents, list):
            logger.warning("FDA cache documents payload invalid for %s/%s/%s", source, drug_id, event_id)
            return []
        docs = [EvidenceDocument.model_validate(item) for item in documents]
        return [self._mark_cached(doc) for doc in docs]

    @staticmethod
    def _mark_cached(doc: EvidenceDocument) -> EvidenceDocument:
        cached = doc.model_copy(deep=True)
        if cached.snippet:
            if _CACHE_NOTE not in cached.snippet:
                cached.snippet = f"{_CACHE_NOTE} {cached.snippet}"
        elif _CACHE_NOTE not in cached.title:
            cached.title = f"{_CACHE_NOTE} {cached.title}"
        return cached

    def _find_event_in_label(
        self,
        label: dict[str, Any],
        event_terms: list[str],
    ) -> tuple[str | None, str | None]:
        normalized_terms = [
            term.strip().lower()
            for term in (event_terms or [])
            if isinstance(term, str) and term.strip()
        ]
        for section in _LABEL_SECTIONS:
            section_content = label.get(section)
            if not section_content:
                continue

            if isinstance(section_content, list):
                text = " ".join(str(item) for item in section_content if item is not None)
            elif isinstance(section_content, dict):
                text = " ".join(str(value) for value in section_content.values() if value is not None)
            else:
                text = str(section_content)

            text_lower = text.lower()
            for event_term in sorted(normalized_terms, key=len, reverse=True):
                if event_term in text_lower:
                    idx = text_lower.find(event_term)
                    start = max(0, idx - 200)
                    end = min(len(text), idx + 300)
                    snippet = text[start:end].strip()
                    return section, snippet

        return None, None

    def _retrieve_safety_comms(
        self,
        drug_id: str,
        event_id: str,
        drug_name: str,
    ) -> list[EvidenceDocument]:
        """
        Scrape FDA Drug Safety Communications page for drug mentions.

        The page is HTML; we search for anchors mentioning the drug name.
        Returns EvidenceDocument objects for matched communications.
        """
        try:
            resp = requests.get(
                self._cfg.safety_comms_url,
                timeout=20,
                headers={
                    "User-Agent": "ClinicalSafetyIntelligence/0.1 (research pipeline; not for commercial use)",
                },
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise EvidenceRetrievalError(f"FDA Safety Comms page unavailable: {exc}") from exc
        return self._parse_safety_comms_html(resp.text, drug_id, event_id, drug_name)

    def _parse_safety_comms_html(
        self,
        html: str,
        drug_id: str,
        event_id: str,
        drug_name: str,
    ) -> list[EvidenceDocument]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("beautifulsoup4 not installed — skipping safety comms scrape.")
            return []

        soup = BeautifulSoup(html, "html.parser")
        drug_lower = drug_name.lower()
        docs: list[EvidenceDocument] = []
        seen_urls: set[str] = set()

        for link in soup.find_all("a", href=True):
            text = link.get_text(strip=True)
            href = link["href"]
            if drug_lower not in text.lower():
                continue
            if href in seen_urls:
                continue

            seen_urls.add(href)
            if href.startswith("/"):
                href = "https://www.fda.gov" + href

            doc_id = f"fda_safety_comm_{drug_id}_{len(docs)}"
            docs.append(
                EvidenceDocument(
                    doc_id=doc_id,
                    source_type="fda_safety_comm",
                    title=text,
                    url=href,
                    identifier=None,
                    publication_date=None,
                    retrieved_date=datetime.now(UTC),
                    snippet=None,
                    credibility=CredibilityClass.REGULATORY,
                    drug_id=drug_id,
                    event_id=event_id,
                    accepted=True,
                )
            )
            if len(docs) >= 5:
                break

        logger.debug("FDA safety comms: found %d communications for '%s'", len(docs), drug_name)
        return docs
