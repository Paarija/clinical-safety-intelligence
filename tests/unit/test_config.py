from __future__ import annotations

import importlib
import os
import sys

import dotenv.main
import pytest

from clinical_safety.common.config import (
    AppConfig,
    DataSourcesConfig,
    DrugScopeConfig,
    EventScopeConfig,
    ModelProvidersConfig,
    SignalThresholdsConfig,
)
from clinical_safety.common.exceptions import ConfigError


def _config() -> AppConfig:
    return AppConfig(
        data_sources=DataSourcesConfig(),
        drug_scope=DrugScopeConfig(),
        event_scope=EventScopeConfig(),
        signal_thresholds=SignalThresholdsConfig(),
        model_providers=ModelProvidersConfig(),
    )


def test_runtime_validation_accepts_default_config() -> None:
    _config().validate_runtime()


def test_runtime_validation_rejects_invalid_page_size() -> None:
    cfg = _config()
    cfg.data_sources.clinicaltrials.page_size = 0

    with pytest.raises(ConfigError, match="clinicaltrials.page_size must be positive"):
        cfg.validate_runtime()


def test_runtime_validation_rejects_invalid_signal_threshold() -> None:
    cfg = _config()
    cfg.signal_thresholds.signal_detection.ci_coverage = 1.5

    with pytest.raises(ConfigError, match="ci_coverage must be between 0 and 1"):
        cfg.validate_runtime()


def test_dotenv_loaded_before_paths_read_data_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    data_dir = tmp_path / "data-root"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"DATA_DIR={data_dir}\nCSI_DOTENV_EXISTING=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setenv("CSI_DOTENV_EXISTING", "from-environment")
    monkeypatch.setattr(
        dotenv.main,
        "find_dotenv",
        lambda *args, **kwargs: str(env_file),
    )

    sys.modules.pop("clinical_safety.common.paths", None)
    sys.modules.pop("clinical_safety.common.config", None)
    loaded_paths = importlib.import_module("clinical_safety.common.paths")

    assert loaded_paths.Paths().data == data_dir.resolve()
    assert os.environ["CSI_DOTENV_EXISTING"] == "from-environment"
