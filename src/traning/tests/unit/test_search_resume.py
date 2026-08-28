"""验证参数搜索不会因重启丢失历史或重复已经完成的 proposal。"""

from __future__ import annotations

import json

import pytest

from traning.config import V2Config
from traning.infrastructure import IntegrityError, SchemaMismatchError
from traning.training import (
    ParameterVector,
    SearchHistoryStore,
    TrialAcceptance,
    TrialObservation,
    run_search,
)


def _initial() -> ParameterVector:
    return ParameterVector(0.001, 0.05, 64, 0.1, 0.0, 0.0)


def _observation(
    trial_index: int,
    parameters: ParameterVector,
    *,
    passed: bool,
) -> TrialObservation:
    return TrialObservation(
        trial_index=trial_index,
        parameters=parameters,
        objective=float(trial_index),
        acceptance=TrialAcceptance(*(passed for _ in range(7))),
    )


class _PassesAtIndex:
    def __init__(self, passing_index: int) -> None:
        self.passing_index = passing_index
        self.calls: list[tuple[int, ParameterVector]] = []

    def evaluate(
        self,
        parameters: ParameterVector,
        trial_index: int,
    ) -> TrialObservation:
        self.calls.append((trial_index, parameters))
        return _observation(
            trial_index,
            parameters,
            passed=trial_index == self.passing_index,
        )


def test_run_search_resumes_after_committed_history_without_repeating() -> None:
    """恢复后首个 evaluator 调用必须从下一个连续 trial 开始。"""

    first = _PassesAtIndex(99)
    committed: list[tuple[TrialObservation, ...]] = []

    def _commit_then_interrupt(history: tuple[TrialObservation, ...]) -> None:
        committed.append(history)
        raise RuntimeError("interrupt")

    with pytest.raises(RuntimeError):
        # 回调模拟进程在第一个原子提交完成后被外部中断。
        run_search(
            first,
            _initial(),
            seed=7,
            on_trial_completed=_commit_then_interrupt,
        )

    assert len(committed) == 1
    resumed = _PassesAtIndex(2)
    snapshots: list[tuple[TrialObservation, ...]] = []
    result = run_search(
        resumed,
        _initial(),
        seed=7,
        history=committed[-1],
        on_trial_completed=snapshots.append,
    )

    assert result.trial_index == 2
    assert tuple(index for index, _parameters in resumed.calls) == (1, 2)
    assert tuple(len(history) for history in snapshots) == (2, 3)


def test_search_history_store_round_trips_and_rejects_tampering(tmp_path) -> None:
    """状态文件必须同时校验历史摘要与 run/data/config 身份。"""

    path = tmp_path / "search-state.json"
    store = SearchHistoryStore(
        path,
        run_id="run-1",
        dataset_id="dataset-1",
        config=V2Config(),
        initial_parameters=_initial(),
    )
    history = (_observation(0, _initial(), passed=False),)
    store.persist(history)
    assert store.load() == history

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["history"][0]["objective"] = 99.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IntegrityError, match="SHA-256"):
        store.load()


def test_search_history_store_rejects_cross_run_resume(tmp_path) -> None:
    """相同路径也不能把另一运行的搜索进度接到当前运行。"""

    path = tmp_path / "search-state.json"
    original = SearchHistoryStore(
        path,
        run_id="run-1",
        dataset_id="dataset-1",
        config=V2Config(),
        initial_parameters=_initial(),
    )
    original.persist((_observation(0, _initial(), passed=False),))
    other = SearchHistoryStore(
        path,
        run_id="run-2",
        dataset_id="dataset-1",
        config=V2Config(),
        initial_parameters=_initial(),
    )

    with pytest.raises(SchemaMismatchError, match="不一致"):
        other.load()
