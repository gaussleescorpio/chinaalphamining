from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from factorforge.contracts import CandidateRecord
from factorforge.evaluation.crossfit import CandidateEvidence


class ArtifactWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "candidate_values").mkdir(exist_ok=True)

    def write_values(
        self,
        record: CandidateRecord,
        evidence: CandidateEvidence,
        timestamps: pd.DatetimeIndex,
        assets: tuple[str, ...],
    ) -> Path:
        time = np.repeat(timestamps.to_numpy(), len(assets))
        asset = np.tile(np.asarray(assets, dtype=object), len(timestamps))
        frame = pd.DataFrame(
            {
                "timestamp": time,
                "asset": asset,
                "candidate_id": record.candidate_id,
                "formula": record.formula_json,
                "atoms": json.dumps(record.atoms, ensure_ascii=False),
                "direction": evidence.final_direction,
                "raw_value": evidence.raw_values.reshape(-1),
                "rank": evidence.rank_values.reshape(-1),
                "score": evidence.score_values.reshape(-1),
            }
        )
        path = self.root / "candidate_values" / f"{record.candidate_id}.parquet"
        frame.to_parquet(path, index=False)
        return path

    def write_json(self, name: str, payload: object) -> Path:
        path = self.root / name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def write_manifest(self) -> Path:
        rows = []
        target = self.root / "SHA256_MANIFEST.csv"
        for path in sorted(
            item for item in self.root.rglob("*") if item.is_file() and item != target
        ):
            rows.append(
                {
                    "relative_path": path.relative_to(self.root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        pd.DataFrame(rows).to_csv(target, index=False, encoding="utf-8-sig")
        return target


def input_fingerprint(
    timestamps: pd.DatetimeIndex,
    assets: tuple[str, ...],
    fields: dict | object,
    labels: np.ndarray,
) -> dict[str, object]:
    digest = hashlib.sha256()
    digest.update(np.asarray(timestamps.asi8, dtype="<i8").tobytes())
    digest.update("\n".join(assets).encode("utf-8"))
    field_hashes = {}
    for name in sorted(fields):
        values = np.asarray(fields[name], dtype="<f8")
        field_digest = hashlib.sha256(values.tobytes()).hexdigest()
        field_hashes[name] = field_digest
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(field_digest))
    label_hash = hashlib.sha256(np.asarray(labels, dtype="<f8").tobytes()).hexdigest()
    digest.update(bytes.fromhex(label_hash))
    return {
        "rows": len(timestamps),
        "assets": len(assets),
        "field_sha256": field_hashes,
        "label_sha256": label_hash,
        "complete_input_sha256": digest.hexdigest(),
    }


def software_environment() -> dict[str, str]:
    import numpy
    import pandas
    import pyarrow
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__,
        "factorforge": "1.0.0",
    }
