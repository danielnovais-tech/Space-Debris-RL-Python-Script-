from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IntegrityResult:
    ok: bool
    reason: str | None = None


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelIntegrityGuard:
    """Simple file-based integrity check for saved models.

    This treats the on-disk model artifact as the 'known good checkpoint'.
    """

    def __init__(self, model_zip_path: str | Path):
        self.model_zip_path = Path(model_zip_path)
        self._baseline: str | None = None

    def establish_baseline(self) -> str:
        self._baseline = sha256_file(self.model_zip_path)
        return self._baseline

    def verify(self) -> IntegrityResult:
        if self._baseline is None:
            self.establish_baseline()
        try:
            current = sha256_file(self.model_zip_path)
        except FileNotFoundError:
            return IntegrityResult(False, "model_missing")

        if current != self._baseline:
            return IntegrityResult(False, "model_hash_mismatch")
        return IntegrityResult(True, None)
