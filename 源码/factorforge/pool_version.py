from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


def build_pool_version(
    selected_ids: Sequence[str],
    directions: Mapping[str, int],
    configuration: Mapping[str, object],
    input_sha256: str,
    parent_version: str | None = None,
) -> dict[str, object]:
    content = {
        "parent_version": parent_version,
        "selected_candidate_ids": list(selected_ids),
        "directions": dict(sorted(directions.items())),
        "configuration": configuration,
        "input_sha256": input_sha256,
    }
    version = hashlib.sha256(
        json.dumps(content, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"factor_pool_version": version, **content}


def read_parent_version(path: str | None) -> str | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("factor_pool_version")
    if not version:
        raise ValueError("parent factor pool file has no factor_pool_version")
    return str(version)
