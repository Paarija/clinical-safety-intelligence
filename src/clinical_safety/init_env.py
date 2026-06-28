"""Local bootstrap helper for first-run development setup."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from clinical_safety.common.config import get_config
from clinical_safety.common.exceptions import ConfigError
from clinical_safety.common.paths import Paths


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
ENV_FILE = PROJECT_ROOT / ".env"


def _copy_env_example() -> bool:
    """Create .env from .env.example when it is absent.

    Returns True when a new file was created. Existing .env files are never
    overwritten because they may contain local secrets.
    """
    if ENV_FILE.exists():
        return False
    if not ENV_EXAMPLE.exists():
        raise FileNotFoundError(f"Missing template: {ENV_EXAMPLE}")
    shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
    return True


def bootstrap() -> bool:
    """Create local bootstrap artifacts and validate runtime config.

    Returns True when .env was created during this run.
    """
    env_created = _copy_env_example()
    Paths().ensure_all()
    get_config().validate_runtime()
    return env_created


def main() -> int:
    try:
        env_created = bootstrap()
    except (ConfigError, OSError) as exc:
        print(f"Bootstrap failed: {exc}", file=sys.stderr)
        return 1

    if env_created:
        print("Created .env from .env.example with blank secret placeholders.")
    else:
        print("Kept existing .env; no values were overwritten.")
    print("Created required local data/report directories.")
    print("Validated local configuration.")
    print("Next: edit .env and set GEMINI_API_KEY for real Gemini synthesis.")
    print("Use --dry-run workflows only when you want placeholder synthesis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
