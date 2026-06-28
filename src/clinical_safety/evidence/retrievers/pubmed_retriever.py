"""
evidence/retrievers/pubmed_retriever.py

Retrieves peer-reviewed literature evidence from NCBI PubMed for a
given drug + adverse event signal.

Two-step NCBI E-utilities workflow:
  1. esearch  — search for PMIDs matching drug+event query
  2. efetch   — retrieve PubMed XML for those PMIDs

Returns a list of EvidenceDocument objects (one per article).

Important:
  - Uses the NCBI free rate limit unless NCBI_API_KEY is configured.
  - Abstracts > 2000 chars are truncated to a snippet.
  - API/network failures raise EvidenceRetrievalError.

Usage:
    from clinical_safety.evidence.retrievers.pubmed_retriever import PubMedRetriever
    retriever = PubMedRetriever()
    docs = retriever.retrieve(drug_id="semaglutide", event_id="pancreatitis",
                              drug_name="semaglutide", event_name="pancreatitis")
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta

import requests

from clinical_safety.common.config import get_config
from clinical_safety.common.exceptions import EvidenceRetrievalError
from clinical_safety.common.logging import get_logger
from clinical_safety.common.paths import Paths
from clinical_safety.common.types import CredibilityClass, EvidenceDocument

logger = get_logger(__name__)

_SNIPPET_MAX_CHARS = 2000
_USER_AGENT = "clinical-safety-intelligence/0.1.0 (research pipeline)"
_CACHE_TTL = timedelta(days=7)
_CACHE_NOTE = "[Retrieved from PubMed cache after API failure; verify source freshness.]"



class PubMedRetriever:
    """
    Retrieves PubMed literature evidence via NCBI E-utilities.

    Returns EvidenceDocument objects sorted by publication date descending.
    """

    def __init__(self, paths: Paths | None = None) -> None:
        cfg = get_config()
        self._cfg = cfg.data_sources.pubmed
        self._delay = self._cfg.request_delay_sec
        self._max_results = self._cfg.max_results_per_query
        self._years_back = self._cfg.years_back
        self._api_key = os.getenv("NCBI_API_KEY")  # None if not set
        self._cache_dir = (paths or Paths()).external / "cache" / "pubmed"

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
        Search PubMed for drug+event literature and return documents.

        Transient NCBI failures fall back to fresh cached evidence when present.
        """
        event_terms = self._normalize_retrieval_terms(event_terms, event_name)
        query = self._build_query(drug_name, event_terms)
        logger.info("PubMed search: '%s'", query)

        try:
            pmids = self._esearch(query)
            if not pmids:
                logger.info("PubMed: no results for query '%s'", query)
                return []

            logger.info("PubMed: %d PMIDs found — fetching abstracts ...", len(pmids))
            docs = self._efetch(pmids, drug_id, event_id)
            self._write_cache(drug_id, event_id, docs)
            logger.info("PubMed: %d documents retrieved", len(docs))
            return docs
        except EvidenceRetrievalError as exc:
            cached_docs = self._read_cache(drug_id, event_id)
            if cached_docs:
                logger.warning("PubMed retrieval failed; using cached documents: %s", exc)
                return cached_docs
            raise

    def _build_query(self, drug_name: str, event_terms: list[str]) -> str:
        """
        Build a PubMed query string.

        Uses MeSH-compatible query with date restriction:
          (drug[tiab] OR drug[MeSH Terms]) AND
          ((term1[tiab] OR term1[MeSH Terms]) OR (term2[tiab] OR term2[MeSH Terms])) AND
          (last N years[pdat])
        """
        min_year = date.today().year - self._years_back
        event_clauses = [
            f'("{term}"[tiab] OR "{term}"[MeSH Terms])' for term in event_terms
        ]
        return (
            f'("{drug_name}"[tiab] OR "{drug_name}"[MeSH Terms]) '
            f'AND ({ " OR ".join(event_clauses) }) '
            f'AND ("{min_year}"[pdat]:"3000"[pdat])'
        )


    def _esearch(self, query: str) -> list[str]:
        """
        Call esearch to get a list of PMIDs.

        Raises EvidenceRetrievalError on API/network failure.
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": self._max_results,
            "retmode": "json",
            "usehistory": "n",
        }
        if self._api_key:
            params["api_key"] = self._api_key
        try:
            resp = requests.get(
                self._cfg.esearch_url,
                params=params,
                timeout=(self._cfg.request_connect_timeout_sec, self._cfg.request_read_timeout_sec),
                headers={"User-Agent": _USER_AGENT},
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise EvidenceRetrievalError(f"PubMed esearch failed: {exc}") from exc

        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        return pmids

    def _efetch(
        self, pmids: list[str], drug_id: str, event_id: str
    ) -> list[EvidenceDocument]:
        """
        Fetch full records for a list of PMIDs in a single request.
        Parses PubMed XML to extract title, abstract, and publication date.
        """
        # efetch accepts comma-separated PMIDs
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self._api_key:
            params["api_key"] = self._api_key
        time.sleep(self._delay)
        try:
            resp = requests.get(
                self._cfg.efetch_url,
                params=params,
                timeout=(self._cfg.request_connect_timeout_sec, self._cfg.request_read_timeout_sec),
                headers={"User-Agent": _USER_AGENT},
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise EvidenceRetrievalError(f"PubMed efetch failed: {exc}") from exc

        return self._parse_pubmed_xml(resp.text, drug_id, event_id)

    def _parse_pubmed_xml(
        self, xml_text: str, drug_id: str, event_id: str
    ) -> list[EvidenceDocument]:
        """Parse PubMed XML response into EvidenceDocument objects."""
        docs: list[EvidenceDocument] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("PubMed XML parse error: %s", exc)
            return []

        for article in root.findall(".//PubmedArticle"):
            try:
                doc = self._parse_single_article(article, drug_id, event_id)
                if doc:
                    docs.append(doc)
            except Exception as exc:
                logger.debug("Skipping article — parse error: %s", exc)
                continue

        # Sort newest first
        docs.sort(key=lambda d: d.publication_date or date.min, reverse=True)
        return docs

    def _cache_path(self, drug_id: str, event_id: str):
        key = hashlib.sha256(f"pubmed|{drug_id}|{event_id}".encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.json"

    def _write_cache(
        self,
        drug_id: str,
        event_id: str,
        docs: list[EvidenceDocument],
    ) -> None:
        if not docs:
            return
        path = self._cache_path(drug_id, event_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "cached_at": datetime.now(UTC).isoformat(),
                "source": "pubmed",
                "drug_id": drug_id,
                "event_id": event_id,
                "documents": [doc.model_dump(mode="json") for doc in docs],
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("PubMed cache write failed for %s/%s: %s", drug_id, event_id, exc)

    def _read_cache(
        self,
        drug_id: str,
        event_id: str,
    ) -> list[EvidenceDocument]:
        path = self._cache_path(drug_id, event_id)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(payload["cached_at"])
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=UTC)
            documents = payload["documents"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("PubMed cache read failed for %s/%s: %s", drug_id, event_id, exc)
            return []
        except ValueError as exc:
            logger.warning("PubMed cache timestamp invalid for %s/%s: %s", drug_id, event_id, exc)
            return []
        if datetime.now(UTC) - cached_at > _CACHE_TTL:
            logger.warning("PubMed cache expired for %s/%s", drug_id, event_id)
            return []
        if not isinstance(documents, list):
            logger.warning("PubMed cache documents payload invalid for %s/%s", drug_id, event_id)
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

    @staticmethod
    def _parse_single_article(
        article: ET.Element, drug_id: str, event_id: str
    ) -> EvidenceDocument | None:
        """Extract fields from one <PubmedArticle> XML element."""
        # PMID
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None
        if not pmid:
            return None

        # Title
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "Untitled"

        # Abstract
        abstract_texts = [
            el.text.strip()
            for el in article.findall(".//AbstractText")
            if el.text
        ]
        abstract = " ".join(abstract_texts)
        snippet = abstract[:_SNIPPET_MAX_CHARS] if abstract else None

        # Publication date (best-effort: year/month/day from PubDate)
        pub_date: date | None = None
        pub_date_el = article.find(".//PubDate")
        if pub_date_el is not None:
            year_el = pub_date_el.find("Year")
            month_el = pub_date_el.find("Month")
            if year_el is not None and year_el.text:
                try:
                    year = int(year_el.text)
                    month = _month_str_to_int(month_el.text if month_el is not None else "1")
                    pub_date = date(year, month, 1)
                except (ValueError, TypeError):
                    pass

        return EvidenceDocument(
            doc_id=f"pubmed_{pmid}",
            source_type="pubmed",
            title=title,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            identifier=pmid,
            publication_date=pub_date,
            retrieved_date=datetime.now(UTC),
            snippet=snippet,
            credibility=CredibilityClass.PEER_REVIEWED,
            drug_id=drug_id,
            event_id=event_id,
            accepted=True,
        )


def _month_str_to_int(month_str: str) -> int:
    """Convert abbreviated month name or number string to int."""
    _MAP = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    try:
        return int(month_str)
    except (ValueError, TypeError):
        return _MAP.get(str(month_str).lower()[:3], 1)
