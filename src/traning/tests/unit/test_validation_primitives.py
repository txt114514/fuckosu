"""集中基础校验只在外部边界收敛值，并保持错误语义明确。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pytest

from traning.lib.validation import (
    require_bool,
    require_enum,
    require_finite,
    require_int,
    require_non_empty_str,
    require_path,
    require_real,
)


class _Mode(str, Enum):
    TRAIN = "train"
    EVAL = "eval"


def test_integer_and_real_validation_reject_boolean_and_non_finite_values() -> None:
    """布尔值不能借由 Python 继承关系冒充数值。"""

    assert require_int(3, "epochs", minimum=1, maximum=4) == 3
    assert require_real(3, "weight", minimum=0.0) == 3.0
    assert require_finite(-2.5, "offset") == -2.5
    with pytest.raises(TypeError, match="epochs"):
        require_int(True, "epochs")
    with pytest.raises(TypeError, match="weight"):
        require_real(False, "weight")
    with pytest.raises(ValueError, match="finite"):
        require_finite(float("nan"), "offset")
    with pytest.raises(ValueError, match="greater than or equal"):
        require_int(0, "epochs", minimum=1)


def test_boolean_string_enum_and_path_validation(tmp_path: Path) -> None:
    """字符串、枚举和路径在进入 typed config 时只规范化一次。"""

    file_path = tmp_path / "config.yaml"
    file_path.write_text("schema_version: 1\n", encoding="utf-8")
    assert require_bool(True, "enabled") is True
    assert require_non_empty_str("run-1", "run_id") == "run-1"
    assert require_enum("train", _Mode, "mode") is _Mode.TRAIN
    assert require_enum(_Mode.EVAL, _Mode, "mode") is _Mode.EVAL
    assert (
        require_path(file_path, "config", must_exist=True, directory=False) == file_path
    )

    with pytest.raises(TypeError, match="enabled"):
        require_bool(1, "enabled")
    with pytest.raises(ValueError, match="must not be empty"):
        require_non_empty_str("   ", "run_id")
    with pytest.raises(ValueError, match="_Mode"):
        require_enum("unknown", _Mode, "mode")
    with pytest.raises(ValueError, match="existing directory"):
        require_path(file_path, "config", directory=True)
