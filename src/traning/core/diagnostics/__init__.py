"""训练诊断与 oracle 上限评估工具的公开入口。"""

from traning.core.diagnostics.oracle_ladder import (
    OracleDiagnosticsResult,
    run_oracle_diagnostics,
)

__all__ = [
    "OracleDiagnosticsResult",
    "run_oracle_diagnostics",
]
