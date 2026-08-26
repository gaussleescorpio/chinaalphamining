from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = ("numpy", "pandas", "pyarrow", "scipy", "psutil", "matplotlib")


def main() -> int:
    missing = []
    for package in REQUIRED:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    if missing:
        print("缺少依赖：" + "、".join(missing))
        print("可运行：python -m pip install -e .")
        return 1
    result = subprocess.run(
        [sys.executable, str(ROOT / "运行因子挖掘.py"), "环境检查"],
        cwd=ROOT,
        check=False,
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
