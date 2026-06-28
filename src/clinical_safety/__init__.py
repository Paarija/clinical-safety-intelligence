from __future__ import annotations

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)

__version__ = "0.1.0"

__all__ = ["__version__"]
