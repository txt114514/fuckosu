"""验证顶层模块登记与 V2 canonical 数据质量启动检查。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from package import DataSplit
from start.checks import run_startup_checks, run_training_startup_checks
from start.modules import source_module_entry, source_module_entries
from traning.config import RuntimeConfig, RuntimeDevice, load_v2_config
from traning.contracts import DataQualityReport


def test_src_module_entries_are_importable_and_point_to_v2_app() -> None:
    keys = {entry.key for entry in source_module_entries(include_start=True)}

    assert {"start", "package", "before_traning", "traning"} <= keys
    assert source_module_entry("traning").importable
    assert source_module_entry("traning").public_entry == "traning.app"


def test_global_startup_checks_pass_without_cuda_requirement() -> None:
    report = run_startup_checks(require_cuda=False)

    assert report.ok
    assert "environment" in {item.key for item in report.results}


def test_training_checks_consume_the_given_quality_report() -> None:
    loaded = load_v2_config(Path("configs/traning.yaml"))
    config = replace(
        loaded,
        runtime=RuntimeConfig(RuntimeDevice.CPU, require_cuda=False, amp=False),
    )
    quality = DataQualityReport(issues=())
    result = run_training_startup_checks(
        config,
        split=DataSplit.TRAIN,
        requested_device=RuntimeDevice.CPU,
        quality_report=quality,
        executor_available=True,
    )

    assert result.ok
    assert result.data_quality is quality
    keys = tuple(item.key for item in result.results)
    assert "training:data_quality" in keys
    assert len(keys) == len(set(keys))
