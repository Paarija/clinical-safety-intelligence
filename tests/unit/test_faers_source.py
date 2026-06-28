from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import clinical_safety.acquisition.faers_source as faers_source
from clinical_safety.acquisition.faers_source import FAERSSource
from clinical_safety.acquisition.source_manifest import SourceManifest
from clinical_safety.common.exceptions import DataSourceError, SourceFileNotFoundError
from clinical_safety.common.paths import Paths


FAERS_DOWNLOAD_URL = "https://example.test/faers_ascii_2026q1.zip"

@pytest.fixture(autouse=True)
def _isolate_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_DIR", raising=False)


class _StreamingResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __enter__(self) -> "_StreamingResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> object:
        assert chunk_size > 0
        yield from self._chunks


def _tiny_faers_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("ascii/DEMO26Q1.TXT", "primaryid$caseid$caseversion\n1$1$1\n")
        zf.writestr("ascii/DRUG26Q1.TXT", "primaryid$drug_seq$role_cod$drugname\n1$1$PS$TESTDRUG\n")
        zf.writestr("ascii/REAC26Q1.TXT", "primaryid$pt\n1$TEST EVENT\n")
        zf.writestr("ascii/OUTC26Q1.TXT", "primaryid$outc_cod\n1$HO\n")
    return buffer.getvalue()


def _patch_config(
    monkeypatch: pytest.MonkeyPatch,
    zip_path: Path,
    *,
    auto_download: bool,
    download_url: str = FAERS_DOWNLOAD_URL,
) -> None:
    faers_cfg = SimpleNamespace(
        zip_path=str(zip_path),
        quarter="2026Q1",
        file_prefixes={},
        encoding="latin-1",
        delimiter="$",
        download_url=download_url,
        auto_download=auto_download,
        request_connect_timeout_sec=2.5,
        request_read_timeout_sec=9.5,
    )
    cfg = SimpleNamespace(data_sources=SimpleNamespace(faers=faers_cfg))
    monkeypatch.setattr(faers_source, "get_config", lambda: cfg)


def _patch_requests_get(monkeypatch: pytest.MonkeyPatch, fake_get: object) -> None:
    requests_module = SimpleNamespace(
        get=fake_get,
        RequestException=requests.RequestException,
        exceptions=requests.exceptions,
    )
    monkeypatch.setattr(faers_source, "requests", requests_module, raising=False)


def _source(data_dir: Path, zip_path: Path | None = None) -> FAERSSource:
    paths = Paths(data_dir=data_dir)
    return FAERSSource(zip_path=zip_path, paths=paths, manifest=SourceManifest(paths))


def test_missing_zip_auto_downloads_and_extracts_required_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = tmp_path / "faers_ascii_2026q1.zip"
    zip_bytes = _tiny_faers_zip()
    calls: list[dict[str, object]] = []

    def fake_get(url: str, **kwargs: object) -> _StreamingResponse:
        calls.append({"url": url, **kwargs})
        return _StreamingResponse([zip_bytes[:8], b"", zip_bytes[8:]])

    _patch_config(monkeypatch, zip_path, auto_download=True)
    _patch_requests_get(monkeypatch, fake_get)

    found = _source(tmp_path / "data").acquire()

    assert zip_path.read_bytes() == zip_bytes
    assert not Path(f"{zip_path}.part").exists()
    assert len(calls) == 1
    assert calls[0]["url"] == FAERS_DOWNLOAD_URL
    assert calls[0]["stream"] is True
    assert calls[0]["timeout"] == (2.5, 9.5)
    assert "clinical-safety-intelligence" in calls[0]["headers"]["User-Agent"]
    assert set(found) == {"DEMO", "DRUG", "REAC", "OUTC"}
    assert found["DEMO"].name == "DEMO26Q1.TXT"
    assert found["DRUG"].read_text(encoding="utf-8").startswith("primaryid$drug_seq")
    manifest_path = tmp_path / "data" / "interim" / "source_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["metadata"]["pipeline_run_id"]
    assert manifest["metadata"]["created_at"]
    assert manifest["metadata"]["saved_at"]
    assert manifest["metadata"]["checksum_algorithm"] == "sha256"
    assert manifest["summary"]["faers"] == 5
    assert len(manifest["entries"]) == 5
    assert all(
        entry["pipeline_run_id"] == manifest["metadata"]["pipeline_run_id"]
        for entry in manifest["entries"]
    )


def test_source_manifest_new_instance_uses_new_run_id_with_existing_entries(
    tmp_path: Path,
) -> None:
    paths = Paths(data_dir=tmp_path / "data")
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"
    first_file.write_text("first", encoding="utf-8")
    second_file.write_text("second", encoding="utf-8")

    first = SourceManifest(paths)
    first.register("faers", first_file)
    first.save()
    first_payload = json.loads(paths.source_manifest.read_text(encoding="utf-8"))
    first_run_id = first_payload["metadata"]["pipeline_run_id"]

    second = SourceManifest(paths)
    second.register("clinicaltrials", second_file)
    second.save()
    second_payload = json.loads(paths.source_manifest.read_text(encoding="utf-8"))
    second_run_id = second_payload["metadata"]["pipeline_run_id"]

    assert second_run_id != first_run_id
    assert second_payload["entries"][0]["pipeline_run_id"] == first_run_id
    assert second_payload["entries"][1]["pipeline_run_id"] == second_run_id



def test_source_manifest_backfills_legacy_entry_run_id(tmp_path: Path) -> None:
    paths = Paths(data_dir=tmp_path / "data")
    paths.source_manifest.parent.mkdir(parents=True, exist_ok=True)
    legacy_file = tmp_path / "legacy.txt"
    legacy_file.write_text("legacy", encoding="utf-8")
    paths.source_manifest.write_text(
        json.dumps([{"source_type": "faers", "path": str(legacy_file)}]),
        encoding="utf-8",
    )

    manifest = SourceManifest(paths)
    assert manifest.entries()[0]["pipeline_run_id"] == "legacy-import"

def test_missing_zip_with_auto_download_disabled_keeps_missing_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = tmp_path / "missing_faers.zip"

    def fail_if_called(*args: object, **kwargs: object) -> object:
        pytest.fail("requests.get should not be called when FAERS auto_download is disabled")

    _patch_config(monkeypatch, zip_path, auto_download=False)
    _patch_requests_get(monkeypatch, fail_if_called)

    with pytest.raises(SourceFileNotFoundError, match="FAERS ZIP not found"):
        _source(tmp_path / "data").acquire()

    assert not zip_path.exists()


def test_failed_download_leaves_no_final_zip_and_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = tmp_path / "faers_ascii_2026q1.zip"

    def fake_get(url: str, **kwargs: object) -> object:
        raise requests.RequestException("connection refused")

    _patch_config(monkeypatch, zip_path, auto_download=True)
    _patch_requests_get(monkeypatch, fake_get)

    with pytest.raises(DataSourceError, match="FAERS.*download.*failed|download.*FAERS.*failed"):
        _source(tmp_path / "data").acquire()

    assert not zip_path.exists()
    assert not Path(f"{zip_path}.part").exists()


def test_empty_download_is_rejected_without_replacing_final_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = tmp_path / "faers_ascii_2026q1.zip"

    def fake_get(url: str, **kwargs: object) -> _StreamingResponse:
        return _StreamingResponse([b"", b""])

    _patch_config(monkeypatch, zip_path, auto_download=True)
    _patch_requests_get(monkeypatch, fake_get)

    with pytest.raises(DataSourceError, match="empty|0 bytes|no data|invalid"):
        _source(tmp_path / "data").acquire()

    assert not zip_path.exists()
    assert not Path(f"{zip_path}.part").exists()
