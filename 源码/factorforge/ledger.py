from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

from factorforge.contracts import CandidateRecord, DecisionRecord


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ResearchLedger:
    """Append-only research ledger for every generated and tested candidate."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                formula_json TEXT NOT NULL,
                formula_sha256 TEXT NOT NULL UNIQUE,
                atoms_json TEXT NOT NULL,
                family TEXT NOT NULL,
                complexity INTEGER NOT NULL,
                maximum_lookback INTEGER NOT NULL,
                unit TEXT NOT NULL,
                generation_batch TEXT NOT NULL,
                seed INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                reason_code TEXT NOT NULL,
                reason_text TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
            );
            CREATE INDEX IF NOT EXISTS decisions_candidate_stage
                ON decisions(candidate_id, stage);
            """)
        self.connection.commit()

    def add_candidate(self, record: CandidateRecord) -> bool:
        cursor = self.connection.execute(
            """INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                record.candidate_id,
                record.formula_json,
                record.formula_sha256,
                json.dumps(record.atoms, ensure_ascii=False),
                record.family,
                record.complexity,
                record.maximum_lookback,
                record.unit,
                record.generation_batch,
                record.seed,
            ),
        )
        return cursor.rowcount == 1

    def add_candidates(self, records) -> int:
        rows = [
            (
                record.candidate_id,
                record.formula_json,
                record.formula_sha256,
                json.dumps(record.atoms, ensure_ascii=False),
                record.family,
                record.complexity,
                record.maximum_lookback,
                record.unit,
                record.generation_batch,
                record.seed,
            )
            for record in records
        ]
        before = self.connection.total_changes
        self.connection.executemany(
            "INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?)", rows
        )
        self.connection.commit()
        return self.connection.total_changes - before

    def candidate_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])

    def add_decision(self, decision: DecisionRecord) -> None:
        self.connection.execute(
            """INSERT INTO decisions
               (candidate_id,stage,accepted,reason_code,reason_text,metrics_json)
               VALUES (?,?,?,?,?,?)""",
            (
                decision.candidate_id,
                decision.stage,
                int(decision.accepted),
                decision.reason_code,
                decision.reason_text,
                json.dumps(
                    _json_safe(decision.metrics),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                ),
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def has_decision(self, candidate_id: str, stage: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM decisions WHERE candidate_id=? AND stage=? LIMIT 1",
            (candidate_id, stage),
        ).fetchone()
        return row is not None

    def stage_results(
        self, stage: str, accepted: bool | None = None
    ) -> dict[str, dict]:
        query = "SELECT candidate_id,accepted,metrics_json FROM decisions WHERE stage=? ORDER BY decision_id"
        rows = self.connection.execute(query, (stage,)).fetchall()
        result = {}
        for candidate_id, flag, metrics in rows:
            if accepted is None or bool(flag) == accepted:
                result[candidate_id] = json.loads(metrics)
        return result

    def stage_count(self, stage: str, accepted: bool | None = None) -> int:
        query = "SELECT COUNT(*) FROM decisions WHERE stage=?"
        parameters: tuple[object, ...] = (stage,)
        if accepted is not None:
            query += " AND accepted=?"
            parameters = (stage, int(accepted))
        return int(self.connection.execute(query, parameters).fetchone()[0])

    def accepted_metric_values(self, stage: str, json_path: str) -> set[str]:
        rows = self.connection.execute(
            "SELECT json_extract(metrics_json, ?) FROM decisions WHERE stage=? AND accepted=1",
            (json_path, stage),
        )
        return {str(row[0]) for row in rows if row[0] is not None}

    def iter_candidates(self, batch_size: int = 256):
        cursor = self.connection.execute(
            """SELECT candidate_id,formula_json,formula_sha256,atoms_json,family,
                      complexity,maximum_lookback,unit,generation_batch,seed
               FROM candidates ORDER BY rowid"""
        )
        while rows := cursor.fetchmany(batch_size):
            yield [
                CandidateRecord(
                    candidate_id=row[0],
                    formula_json=row[1],
                    formula_sha256=row[2],
                    atoms=tuple(json.loads(row[3])),
                    family=row[4],
                    complexity=int(row[5]),
                    maximum_lookback=int(row[6]),
                    unit=row[7],
                    generation_batch=row[8],
                    seed=int(row[9]),
                )
                for row in rows
            ]

    def sample_candidates(self, count: int) -> list[CandidateRecord]:
        rows = self.connection.execute(
            """SELECT candidate_id,formula_json,formula_sha256,atoms_json,family,
                      complexity,maximum_lookback,unit,generation_batch,seed
               FROM candidates ORDER BY formula_sha256 LIMIT ?""",
            (count,),
        ).fetchall()
        return [
            CandidateRecord(
                candidate_id=row[0],
                formula_json=row[1],
                formula_sha256=row[2],
                atoms=tuple(json.loads(row[3])),
                family=row[4],
                complexity=int(row[5]),
                maximum_lookback=int(row[6]),
                unit=row[7],
                generation_batch=row[8],
                seed=int(row[9]),
            )
            for row in rows
        ]

    def iter_candidates_pending_screening(self, batch_size: int = 256):
        cursor = self.connection.execute(
            """SELECT c.candidate_id,c.formula_json,c.formula_sha256,c.atoms_json,c.family,
                      c.complexity,c.maximum_lookback,c.unit,c.generation_batch,c.seed
               FROM candidates c
               WHERE EXISTS (
                   SELECT 1 FROM decisions chosen
                   WHERE chosen.candidate_id=c.candidate_id
                     AND chosen.stage='STRUCTURAL_PREFILTER' AND chosen.accepted=1
               )
                 AND NOT EXISTS (
                   SELECT 1 FROM decisions d
                   WHERE d.candidate_id=c.candidate_id
                     AND (d.stage='SCREENING' OR (d.stage='LABEL_BLIND' AND d.accepted=0))
               )
               ORDER BY c.rowid"""
        )
        while rows := cursor.fetchmany(batch_size):
            yield [
                CandidateRecord(
                    candidate_id=row[0],
                    formula_json=row[1],
                    formula_sha256=row[2],
                    atoms=tuple(json.loads(row[3])),
                    family=row[4],
                    complexity=int(row[5]),
                    maximum_lookback=int(row[6]),
                    unit=row[7],
                    generation_batch=row[8],
                    seed=int(row[9]),
                )
                for row in rows
            ]

    def record_structural_prefilter(
        self, capacity: int, mechanism_quota_per_card: int = 1
    ) -> int:
        """Choose a deterministic, family-covered numeric budget without labels."""

        if self.stage_count("STRUCTURAL_PREFILTER"):
            return self.stage_count("STRUCTURAL_PREFILTER", accepted=True)
        self.connection.execute(
            "CREATE TEMP TABLE structural_choice(candidate_id TEXT PRIMARY KEY)"
        )
        mechanism = self.connection.execute(
            """SELECT candidate_id FROM (
                   SELECT candidate_id,family,
                          ROW_NUMBER() OVER (
                              PARTITION BY family ORDER BY complexity,formula_sha256
                          ) AS family_rank
                   FROM candidates
                   WHERE family LIKE 'mechanism:%'
               ) WHERE family_rank<=? ORDER BY family,candidate_id LIMIT ?""",
            (mechanism_quota_per_card, capacity),
        ).fetchall()
        chosen = [row[0] for row in mechanism]
        remaining_capacity = max(0, capacity - len(chosen))
        family = self.connection.execute(
            """SELECT candidate_id FROM (
                   SELECT candidate_id,family,
                          ROW_NUMBER() OVER (
                              PARTITION BY family ORDER BY complexity,formula_sha256
                          ) AS family_rank
                   FROM candidates
               ) WHERE family_rank=1 LIMIT ?""",
            (remaining_capacity,),
        ).fetchall()
        chosen_set = set(chosen)
        chosen.extend(row[0] for row in family if row[0] not in chosen_set)
        remaining = max(0, capacity - len(chosen))
        if remaining:
            placeholders = ",".join("?" for _ in chosen)
            exclusion = f"WHERE candidate_id NOT IN ({placeholders})" if chosen else ""
            rows = self.connection.execute(
                f"SELECT candidate_id FROM candidates {exclusion} ORDER BY formula_sha256 LIMIT ?",
                (*chosen, remaining),
            ).fetchall()
            chosen.extend(row[0] for row in rows)
        self.connection.executemany(
            "INSERT INTO structural_choice(candidate_id) VALUES (?)",
            ((item,) for item in chosen),
        )
        self.connection.execute(
            """INSERT INTO decisions(candidate_id,stage,accepted,reason_code,reason_text,metrics_json)
               SELECT c.candidate_id,'STRUCTURAL_PREFILTER',
                      CASE WHEN s.candidate_id IS NULL THEN 0 ELSE 1 END,
                      CASE WHEN s.candidate_id IS NULL
                           THEN 'FIXED_NUMERIC_BUDGET_REJECT' ELSE 'FIXED_NUMERIC_BUDGET_PASS' END,
                      'Deterministic label-blind family coverage and hash sampling.',
                      json_object('capacity', ?)
               FROM candidates c LEFT JOIN structural_choice s ON s.candidate_id=c.candidate_id""",
            (capacity,),
        )
        self.connection.commit()
        return len(chosen)

    def screening_shortlist(self, capacity: int) -> list[str]:
        order = """CAST(json_extract(d.metrics_json,'$.positive_block_fraction') AS REAL) DESC,
                   CAST(json_extract(d.metrics_json,'$.worst_block_mean') AS REAL) DESC,
                   CAST(json_extract(d.metrics_json,'$.rank_ic_mean') AS REAL) DESC,
                   CAST(json_extract(d.metrics_json,'$.tail_spread') AS REAL) DESC,
                   CAST(json_extract(d.metrics_json,'$.mean_net_return') AS REAL) DESC,
                   c.candidate_id ASC"""
        family_rows = self.connection.execute(
            f"""SELECT candidate_id FROM (
                    SELECT c.candidate_id,c.family,
                           ROW_NUMBER() OVER (PARTITION BY c.family ORDER BY {order}) AS family_rank
                    FROM candidates c JOIN decisions d ON d.candidate_id=c.candidate_id
                    WHERE d.stage='SCREENING' AND d.accepted=1
                ) WHERE family_rank=1 LIMIT ?""",
            (capacity,),
        ).fetchall()
        chosen = [row[0] for row in family_rows]
        remaining = capacity - len(chosen)
        if remaining <= 0:
            return chosen
        placeholders = ",".join("?" for _ in chosen)
        exclusion = f"AND c.candidate_id NOT IN ({placeholders})" if chosen else ""
        rows = self.connection.execute(
            f"""SELECT c.candidate_id
                FROM candidates c JOIN decisions d ON d.candidate_id=c.candidate_id
                WHERE d.stage='SCREENING' AND d.accepted=1 {exclusion}
                ORDER BY {order} LIMIT ?""",
            (*chosen, remaining),
        ).fetchall()
        return [*chosen, *(row[0] for row in rows)]

    def record_shortlist(self, selected_ids: list[str], capacity: int) -> None:
        self.connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS selected_shortlist(candidate_id TEXT PRIMARY KEY)"
        )
        self.connection.execute("DELETE FROM selected_shortlist")
        self.connection.executemany(
            "INSERT INTO selected_shortlist(candidate_id) VALUES (?)",
            ((item,) for item in selected_ids),
        )
        self.connection.execute(
            """INSERT INTO decisions(candidate_id,stage,accepted,reason_code,reason_text,metrics_json)
               SELECT d.candidate_id,'BOUNDED_SHORTLIST',
                      CASE WHEN s.candidate_id IS NULL THEN 0 ELSE 1 END,
                      CASE WHEN s.candidate_id IS NULL THEN 'SHORTLIST_CAPACITY_REJECT' ELSE 'SHORTLIST_PASS' END,
                      'Fixed stage-one shortlist with family coverage; no outer interval is used.',
                      json_object('capacity', ?)
               FROM decisions d
               LEFT JOIN selected_shortlist s ON s.candidate_id=d.candidate_id
               WHERE d.stage='SCREENING' AND d.accepted=1
                 AND NOT EXISTS (
                     SELECT 1 FROM decisions prior
                     WHERE prior.candidate_id=d.candidate_id AND prior.stage='BOUNDED_SHORTLIST'
                 )""",
            (capacity,),
        )
        self.connection.commit()

    def screening_statistics(self):
        import pandas as pd

        return pd.read_sql_query(
            """SELECT c.candidate_id,c.family,c.formula_sha256,d.accepted,d.reason_code,d.metrics_json
               FROM candidates c JOIN decisions d ON d.candidate_id=c.candidate_id
               WHERE d.stage='SCREENING' ORDER BY c.rowid""",
            self.connection,
        )

    def family_coverage_statistics(self):
        import pandas as pd

        return pd.read_sql_query(
            """SELECT c.family,
                      COUNT(*) AS catalog_count,
                      SUM(CASE WHEN s.stage='SCREENING' THEN 1 ELSE 0 END) AS screening_evaluated,
                      SUM(CASE WHEN s.stage='SCREENING' AND s.accepted=1 THEN 1 ELSE 0 END) AS screening_pass,
                      SUM(CASE WHEN e.stage='DEVELOPMENT_CROSSFIT' AND e.accepted=1 THEN 1 ELSE 0 END) AS exact_evaluated,
                      SUM(CASE WHEN p.stage='INCREMENTAL_POOL' AND p.accepted=1 THEN 1 ELSE 0 END) AS selected
               FROM candidates c
               LEFT JOIN decisions s ON s.candidate_id=c.candidate_id AND s.stage='SCREENING'
               LEFT JOIN decisions e ON e.candidate_id=c.candidate_id AND e.stage='DEVELOPMENT_CROSSFIT'
               LEFT JOIN decisions p ON p.candidate_id=c.candidate_id AND p.stage='INCREMENTAL_POOL'
               GROUP BY c.family ORDER BY c.family""",
            self.connection,
        )

    def get_candidate(self, candidate_id: str) -> CandidateRecord:
        row = self.connection.execute(
            """SELECT candidate_id,formula_json,formula_sha256,atoms_json,family,
                      complexity,maximum_lookback,unit,generation_batch,seed
               FROM candidates WHERE candidate_id=?""",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        return CandidateRecord(
            candidate_id=row[0],
            formula_json=row[1],
            formula_sha256=row[2],
            atoms=tuple(json.loads(row[3])),
            family=row[4],
            complexity=int(row[5]),
            maximum_lookback=int(row[6]),
            unit=row[7],
            generation_batch=row[8],
            seed=int(row[9]),
        )

    def export(self, output: str | Path) -> None:
        import pandas as pd

        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        pd.read_sql_query(
            "SELECT * FROM candidates ORDER BY rowid", self.connection
        ).to_parquet(root / "完整候选目录.parquet", index=False)
        pd.read_sql_query(
            "SELECT * FROM decisions ORDER BY decision_id", self.connection
        ).to_parquet(root / "完整筛选淘汰记录.parquet", index=False)

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> "ResearchLedger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
