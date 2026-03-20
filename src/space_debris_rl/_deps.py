from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MissingDependencyError(ImportError):
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


def require(module: str, extra: str, pip_name: str | None = None) -> None:
    """Raise a helpful error if an optional dependency is missing."""
    try:
        __import__(module)
    except ImportError as exc:  # pragma: no cover
        pkg = pip_name or module
        raise MissingDependencyError(
            f"Missing optional dependency '{pkg}'. Install with: pip install 'space-debris-rl[{extra}]'"
        ) from exc
