"""验证数据质量的 blocking 只由 canonical 字段决定。"""

from __future__ import annotations

import pytest

from traning.contracts import DataQualitySeverity
from traning.data import (
    DataQualityContext,
    DataQualityFinding,
    DataQualityGate,
    DataQualityRule,
    QualityGateBlockedError,
    require_quality,
)


def _finding(_context: DataQualityContext) -> tuple[DataQualityFinding, ...]:
    return (DataQualityFinding(None, "人工质量问题"),)


def test_info_issue_can_block_training() -> None:
    """UI severity 不能覆盖领域层 blocks_training。"""

    gate = DataQualityGate(
        rules=(
            DataQualityRule(
                code="info_but_blocking",
                severity=DataQualitySeverity.INFO,
                blocks_training=True,
                evaluate=_finding,
            ),
        )
    )
    report = gate.evaluate(DataQualityContext.from_samples(()))
    assert report.ok is False
    with pytest.raises(QualityGateBlockedError) as captured:
        require_quality(report)
    assert captured.value.report is report


def test_error_issue_can_be_nonblocking() -> None:
    """ERROR 也不会被 pipeline 擅自解释为 blocking。"""

    gate = DataQualityGate(
        rules=(
            DataQualityRule(
                code="error_but_nonblocking",
                severity=DataQualitySeverity.ERROR,
                blocks_training=False,
                evaluate=_finding,
            ),
        )
    )
    report = gate.evaluate(DataQualityContext.from_samples(()))
    assert report.ok is True
    require_quality(report)


def test_default_gate_blocks_empty_training_split() -> None:
    """默认门禁不允许空训练集继续进入训练。"""

    report = DataQualityGate().evaluate(DataQualityContext.from_samples(()))
    assert report.ok is False
    assert any(
        issue.code == "missing_training_split" and issue.blocks_training
        for issue in report.issues
    )
