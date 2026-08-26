from __future__ import annotations

import json
from pathlib import Path

from factorforge.a_share.cleaning import CleanResult


def write_cleaning_artifacts(result: CleanResult, output_directory: str | Path) -> None:
    """Write a cleaned table and a human/auditor-readable issue report."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result.frame.to_parquet(output / "cleaned_data.parquet", index=False)
    result.issue_table().to_csv(output / "quality_issues.csv", index=False, encoding="utf-8-sig")
    summary = {
        "passed": result.passed,
        "rows": len(result.frame),
        "issue_count": len(result.issues),
        "error_count": sum(item.severity.value == "error" for item in result.issues),
        "warning_count": sum(item.severity.value == "warning" for item in result.issues),
    }
    (output / "quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
