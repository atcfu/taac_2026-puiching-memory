from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_hash64(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    hashed = int.from_bytes(digest, byteorder="big", signed=False)
    return (hashed % ((1 << 63) - 1)) + 1


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


__all__ = ["ensure_dir", "stable_hash64", "write_json"]
