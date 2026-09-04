"""生产 curriculum/ASHA 状态图、摘要与恢复身份验收。"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
from package import CurriculumStage

from traning.conf import OptimizationConfig, V2Config
from traning.lib.infrastructure import IntegrityError, SchemaMismatchError
from traning.core.training import (
    AshaAction,
    ParameterVector,
    ProductionJobRecord,
    ProductionScheduleState,
    ProductionScheduleStore,
    ProductionTrialContext,
    TrialAcceptance,
    TrialObservation,
)


_ARTIFACT_SHA256 = "a" * 64


def _initial() -> ParameterVector:
    """返回处于注册表合法范围内的稳定初始 proposal。"""

    return ParameterVector(0.001, 0.05, 64, 0.1, 0.0, 0.0)


def _acceptance(*, domain_passed: bool, schedule: bool = False) -> TrialAcceptance:
    """构造明确区分领域 gate 与完整调度 gate 的验收结果。"""

    return TrialAcceptance(
        data=domain_passed,
        perception=domain_passed,
        tracking=domain_passed,
        belief=domain_passed,
        outcome=domain_passed,
        decision=domain_passed,
        golden=domain_passed,
        schedule=schedule,
    )


def _store(path: Path, *, config: V2Config | None = None) -> ProductionScheduleStore:
    """创建绑定固定 run/data/initial 身份的测试 store。"""

    return ProductionScheduleStore(
        path,
        run_id="run-1",
        dataset_id="dataset-1",
        config=V2Config() if config is None else config,
        initial_parameters=_initial(),
    )


def _completed_winner_state(store: ProductionScheduleStore) -> ProductionScheduleState:
    """构造逐阶段、逐 rung、checkpoint 连续的完整 winner ledger。"""

    contexts: list[ProductionTrialContext] = []
    jobs: list[ProductionJobRecord] = []
    parent: Path | None = None
    stages = tuple(CurriculumStage)
    for stage_index, stage in enumerate(stages):
        for rung_index, budget_steps in enumerate((1, 4)):
            context = ProductionTrialContext(
                cohort_index=0,
                trial_index=0,
                parameters=_initial(),
                curriculum_stage=stage,
                rung_index=rung_index,
                budget_steps=budget_steps,
                parent_checkpoint_path=parent,
            )
            checkpoint = Path(
                f"artifacts/trial-0/{stage.value}/rung-{rung_index}/checkpoint"
            )
            final = stage is CurriculumStage.FULL and rung_index == 1
            action = AshaAction.PROMOTE if rung_index == 0 else AshaAction.CONTINUE
            acceptance = _acceptance(domain_passed=True, schedule=final)
            contexts.append(context)
            jobs.append(
                ProductionJobRecord(
                    context=context,
                    objective=float(stage_index * 2 + rung_index),
                    gate_passed=True,
                    acceptance=acceptance,
                    action=action,
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=_ARTIFACT_SHA256,
                    feedback_path=Path(
                        f"artifacts/trial-0/{stage.value}/rung-{rung_index}/feedback.json"
                    ),
                    feedback_sha256="b" * 64,
                )
            )
            parent = checkpoint
    final_job = jobs[-1]
    observation = TrialObservation(
        trial_index=0,
        parameters=_initial(),
        objective=final_job.objective,
        acceptance=final_job.acceptance,
    )
    return replace(
        store.empty_state(),
        contexts=tuple(contexts),
        jobs=tuple(jobs),
        history=(observation,),
    )


def test_schedule_store_round_trips_complete_winner_and_payload_sha(
    tmp_path: Path,
) -> None:
    """完整课程 winner 必须无损恢复且摘要覆盖整个 canonical payload。"""

    path = tmp_path / "production-schedule.json"
    store = _store(path)
    committed = store.persist(_completed_winner_state(store))

    assert committed.updated_at_ms > 0.0
    assert store.load() == committed
    root = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        root["payload"],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert root["payload_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert root["payload"]["history"][0]["acceptance"]["schedule"] is True


def test_schedule_store_returns_identity_bound_empty_state(tmp_path: Path) -> None:
    """尚无状态文件时也要返回携带当前身份和 rung 数量的 typed 空状态。"""

    store = _store(tmp_path / "missing.json")

    state = store.load()

    assert state == store.empty_state()
    assert state.initial_parameters == _initial()
    assert state.rung_count == 2
    assert state.next_trial_index == 0
    assert state.tried_parameters == ()


def test_schedule_store_rejects_payload_tampering_and_cross_config_resume(
    tmp_path: Path,
) -> None:
    """摘要损坏和配置身份变化都不能静默恢复旧调度图。"""

    path = tmp_path / "production-schedule.json"
    original = _store(path)
    original.persist(_completed_winner_state(original))
    root = json.loads(path.read_text(encoding="utf-8"))
    root["payload"]["jobs"][0]["objective"] = 999.0
    path.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(IntegrityError, match="SHA-256"):
        original.load()

    original.persist(_completed_winner_state(original))
    changed = _store(
        path,
        config=V2Config(
            optimization=OptimizationConfig(
                asha_rungs=V2Config().optimization.asha_rungs,
                hard_example_bonus=2.0,
            )
        ),
    )
    with pytest.raises(SchemaMismatchError, match="不一致"):
        changed.load()


def test_schedule_store_rejects_unknown_payload_key_and_schema_version(
    tmp_path: Path,
) -> None:
    """即使攻击者重算摘要，未知字段和非活动 schema 仍必须硬失败。"""

    path = tmp_path / "production-schedule.json"
    store = _store(path)
    store.persist(store.empty_state())
    root = json.loads(path.read_text(encoding="utf-8"))
    root["schema_version"] = 999
    path.write_text(json.dumps(root), encoding="utf-8")
    with pytest.raises(SchemaMismatchError, match="仅支持 schema"):
        store.load()

    store.persist(store.empty_state())
    root = json.loads(path.read_text(encoding="utf-8"))
    root["payload"]["unknown"] = True
    canonical = json.dumps(
        root["payload"],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    root["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(root), encoding="utf-8")
    with pytest.raises(SchemaMismatchError, match="payload 字段集合"):
        store.load()


def test_schedule_store_rejects_context_budget_or_cohort_not_from_config(
    tmp_path: Path,
) -> None:
    """context 的 cohort 与累计预算必须来自当前生产配置而非任意调用值。"""

    store = _store(tmp_path / "production-schedule.json")
    context = ProductionTrialContext(
        cohort_index=0,
        trial_index=0,
        parameters=_initial(),
        curriculum_stage=CurriculumStage.BASIC,
        rung_index=0,
        budget_steps=2,
    )
    state = replace(store.empty_state(), contexts=(context,))

    with pytest.raises(SchemaMismatchError, match="累计预算"):
        store.persist(state)


def test_schedule_state_rejects_skipped_transition_and_parent_mismatch() -> None:
    """同一 proposal 不得跳 rung，晋级后也必须消费精确的父 checkpoint。"""

    base = ProductionScheduleState(
        run_id="run-1",
        dataset_id="dataset-1",
        config_sha256="c" * 64,
        initial_parameters=_initial(),
        rung_count=2,
    )
    first = ProductionTrialContext(
        cohort_index=0,
        trial_index=0,
        parameters=_initial(),
        curriculum_stage=CurriculumStage.BASIC,
        rung_index=0,
        budget_steps=1,
    )
    skipped = replace(first, rung_index=1, budget_steps=4)
    with pytest.raises(ValueError, match="必须从 BASIC rung 0"):
        replace(base, contexts=(skipped,))

    first_job = ProductionJobRecord(
        context=first,
        objective=1.0,
        gate_passed=True,
        acceptance=_acceptance(domain_passed=True),
        action=AshaAction.PROMOTE,
        checkpoint_path=Path("checkpoint-a"),
        checkpoint_sha256=_ARTIFACT_SHA256,
    )
    next_context = replace(
        first,
        rung_index=1,
        budget_steps=4,
        parent_checkpoint_path=Path("checkpoint-b"),
    )
    with pytest.raises(ValueError, match="精确引用"):
        replace(
            base,
            contexts=(first, next_context),
            jobs=(first_job,),
        )


def test_schedule_state_rejects_early_schedule_pass_and_duplicate_proposal() -> None:
    """schedule gate 不能在早期 rung 通过，不同 trial 也不能复用参数。"""

    context = ProductionTrialContext(
        cohort_index=0,
        trial_index=0,
        parameters=_initial(),
        curriculum_stage=CurriculumStage.BASIC,
        rung_index=0,
        budget_steps=1,
    )
    early_pass = ProductionJobRecord(
        context=context,
        objective=1.0,
        gate_passed=True,
        acceptance=_acceptance(domain_passed=True, schedule=True),
        action=AshaAction.PROMOTE,
        checkpoint_path=Path("checkpoint"),
        checkpoint_sha256=_ARTIFACT_SHA256,
    )
    base = ProductionScheduleState(
        run_id="run-1",
        dataset_id="dataset-1",
        config_sha256="c" * 64,
        initial_parameters=_initial(),
        rung_count=2,
    )
    with pytest.raises(ValueError, match="schedule gate"):
        replace(base, contexts=(context,), jobs=(early_pass,))

    duplicate = ProductionTrialContext(
        cohort_index=0,
        trial_index=1,
        parameters=_initial(),
        curriculum_stage=CurriculumStage.BASIC,
        rung_index=0,
        budget_steps=1,
    )
    with pytest.raises(ValueError, match="不得重复 ParameterVector"):
        replace(base, contexts=(context, duplicate))


def test_job_artifact_paths_and_hashes_are_strict_pairs() -> None:
    """checkpoint/feedback 引用不能只有路径或只有摘要。"""

    context = ProductionTrialContext(
        cohort_index=0,
        trial_index=0,
        parameters=_initial(),
        curriculum_stage=CurriculumStage.BASIC,
        rung_index=0,
        budget_steps=1,
    )

    with pytest.raises(ValueError, match="同时存在"):
        ProductionJobRecord(
            context=context,
            objective=0.0,
            gate_passed=False,
            acceptance=_acceptance(domain_passed=False),
            checkpoint_path=Path("checkpoint"),
        )
    with pytest.raises(ValueError, match="同时存在"):
        ProductionJobRecord(
            context=context,
            objective=0.0,
            gate_passed=False,
            acceptance=_acceptance(domain_passed=False),
            feedback_sha256=_ARTIFACT_SHA256,
        )
    with pytest.raises(ValueError, match="64 位"):
        ProductionJobRecord(
            context=context,
            objective=0.0,
            gate_passed=False,
            acceptance=_acceptance(domain_passed=False),
            checkpoint_path=Path("checkpoint"),
            checkpoint_sha256="not-a-sha",
        )
