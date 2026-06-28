from __future__ import annotations

from typing import Protocol


class SecretProvider(Protocol):
    """Interface for resolving runtime secrets from an external store."""

    def get_secret(self, name: str) -> str | None:
        """Return the secret value for ``name``, or ``None`` when it is unavailable."""
