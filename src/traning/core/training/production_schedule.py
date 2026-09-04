"""生产 curriculum/ASHA job 的强类型状态与原子恢复边界。"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path
import time

from package import CurriculumStage

from traning.conf import V2Config
from traning.state.common import require_identifier, require_sha256
from traning.lib.infrastructure import (
    IntegrityError,
    SchemaMismatchError,
    atomic_write_json,
    read_json_object,
)
from traning.core.training.optimization import (
    ParameterVector,
    TrialAcceptance,
    TrialObservation,
)
from traning.core.training.scheduling import AshaAction, CURRICULUM_ORDER
from traning.core.training.search_state import training_config_sha256


PRODUCTION_SCHEDULE_SCHEMA_VERSION = 1
"""当前唯一支持的生产调度恢复状态版本。"""

_ROOT_KEYS = frozenset({"schema_version", "payload", "payload_sha256"})
_PAYLOAD_KEYS = frozenset(
    {
        "run_id",
        "dataset_id",
        "config_sha256",
        "initial_parameters",
        "rung_count",
        "contexts",
        "jobs",
        "history",
        "updated_at_ms",
    }
)
_CONTEXT_KEYS = frozenset(
    {
        "cohort_index",
        "trial_index",
        "parameters",
        "curriculum_stage",
        "rung_index",
        "budget_steps",
        "parent_checkpoint_path",
    }
)
_JOB_KEYS = frozenset(
    {
        "context",
        "objective",
        "acceptance",
        "gate_passed",
        "action",
        "checkpoint_path",
        "checkpoint_sha256",
        "feedback_path",
        "feedback_sha256",
    }
)
_OBSERVATION_KEYS = frozenset({"trial_index", "parameters", "objective", "acceptance"})
_PARAMETER_FIELDS = tuple(field.name for field in fields(ParameterVector))


@dataclass(frozen=True, slots=True)
class ProductionTrialContext:
    """一个 proposal 在指定 cohort、课程阶段和 ASHA rung 的累计预算。"""

    cohort_index: int
    trial_index: int
    parameters: ParameterVector
    curriculum_stage: CurriculumStage
    rung_index: int
    budget_steps: int
    parent_checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        for name in ("cohort_index", "trial_index", "rung_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 必须是整数")
            if value < 0:
                raise ValueError(f"{name} 不得为负数")
        if not isinstance(self.parameters, ParameterVector):
            raise TypeError("parameters 必须是 ParameterVector")
        if not isinstance(self.curriculum_stage, CurriculumStage):
            raise TypeError("curriculum_stage 必须是 package.CurriculumStage")
        if isinstance(self.budget_steps, bool) or not isinstance(
            self.budget_steps, int
        ):
            raise TypeError("budget_steps 必须是整数")
        if self.budget_steps < 1:
            raise ValueError("budget_steps 必须是累计正预算")
        _optional_path("parent_checkpoint_path", self.parent_checkpoint_path)

    @property
    def key(self) -> tuple[int, int, CurriculumStage, int]:
        """返回不会把同一 proposal 的多个 rung 混在一起的稳定 job key。"""

        return (
            self.cohort_index,
            self.trial_index,
            self.curriculum_stage,
            self.rung_index,
        )


@dataclass(frozen=True, slots=True)
class ProductionJobRecord:
    """一个已完成 job 的指标、门禁、调度动作和可恢复制品引用。"""

    context: ProductionTrialContext
    objective: float
    acceptance: TrialAcceptance
    gate_passed: bool
    action: AshaAction | None = None
    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None
    feedback_path: Path | None = None
    feedback_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, ProductionTrialContext):
            raise TypeError("context 必须是 ProductionTrialContext")
        if isinstance(self.objective, bool) or not isinstance(
            self.objective, (int, float)
        ):
            raise TypeError("objective 必须是数值")
        if not math.isfinite(float(self.objective)):
            raise ValueError("objective 必须是有限数值")
        if not isinstance(self.acceptance, TrialAcceptance):
            raise TypeError("acceptance 必须是 TrialAcceptance")
        if not isinstance(self.gate_passed, bool):
            raise TypeError("gate_passed 必须是 bool")
        if self.gate_passed != _domain_gates_passed(self.acceptance):
            raise ValueError("gate_passed 必须等于 schedule 之外的全部领域门禁")
        if self.action is not None and not isinstance(self.action, AshaAction):
            raise TypeError("action 必须是 AshaAction 或 None")
        if (
            self.action in (AshaAction.PROMOTE, AshaAction.CONTINUE)
            and not self.gate_passed
        ):
            raise ValueError("领域门禁失败的 job 不得晋级或继续")
        _artifact_pair(
            "checkpoint",
            path=self.checkpoint_path,
            sha256=self.checkpoint_sha256,
        )
        _artifact_pair(
            "feedback",
            path=self.feedback_path,
            sha256=self.feedback_sha256,
        )
        if self.action in (AshaAction.PROMOTE, AshaAction.CONTINUE) and (
            self.checkpoint_path is None
        ):
            raise ValueError("晋级或继续的 job 必须具有 checkpoint artifact")


@dataclass(frozen=True, slots=True)
class ProductionScheduleState:
    """完整 proposal/job/history 调度快照；任何恢复都从该对象继续。"""

    run_id: str
    dataset_id: str
    config_sha256: str
    initial_parameters: ParameterVector
    rung_count: int
    contexts: tuple[ProductionTrialContext, ...] = ()
    jobs: tuple[ProductionJobRecord, ...] = ()
    history: tuple[TrialObservation, ...] = ()
    updated_at_ms: float = 0.0

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run_id")
        require_identifier(self.dataset_id, "dataset_id")
        require_sha256(self.config_sha256)
        if not isinstance(self.initial_parameters, ParameterVector):
            raise TypeError("initial_parameters 必须是 ParameterVector")
        if isinstance(self.rung_count, bool) or not isinstance(self.rung_count, int):
            raise TypeError("rung_count 必须是整数")
        if self.rung_count < 1:
            raise ValueError("rung_count 必须至少为 1")
        if not isinstance(self.contexts, tuple) or any(
            not isinstance(context, ProductionTrialContext) for context in self.contexts
        ):
            raise TypeError("contexts 必须是 ProductionTrialContext 元组")
        if not isinstance(self.jobs, tuple) or any(
            not isinstance(job, ProductionJobRecord) for job in self.jobs
        ):
            raise TypeError("jobs 必须是 ProductionJobRecord 元组")
        if not isinstance(self.history, tuple) or any(
            not isinstance(observation, TrialObservation)
            for observation in self.history
        ):
            raise TypeError("history 必须是 TrialObservation 元组")
        if (
            isinstance(self.updated_at_ms, bool)
            or not isinstance(self.updated_at_ms, (int, float))
            or not math.isfinite(float(self.updated_at_ms))
            or self.updated_at_ms < 0.0
        ):
            raise ValueError("updated_at_ms 必须是有限非负数")
        self._validate_graph()

    @property
    def next_trial_index(self) -> int:
        """返回 proposal ledger 的下一个连续 trial index。"""

        return len({context.trial_index for context in self.contexts})

    @property
    def tried_parameters(self) -> tuple[ParameterVector, ...]:
        """按 trial index 返回每个 proposal 一次，供确定性搜索避免重复。"""

        by_trial: dict[int, ParameterVector] = {}
        for context in self.contexts:
            by_trial.setdefault(context.trial_index, context.parameters)
        return tuple(by_trial[index] for index in sorted(by_trial))

    def _validate_graph(self) -> None:
        context_by_key = _validate_contexts(self)
        jobs_by_key = _validate_jobs(self, context_by_key)
        _validate_transitions(self, jobs_by_key)
        _validate_history(self, jobs_by_key)


class ProductionScheduleStore:
    """以完整 payload 摘要和 manifest-last 原子写保存生产调度状态。"""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        dataset_id: str,
        config: V2Config,
        initial_parameters: ParameterVector,
    ) -> None:
        if not isinstance(path, Path):
            raise TypeError("path 必须是 pathlib.Path")
        if not isinstance(config, V2Config):
            raise TypeError("config 必须是 V2Config")
        if not isinstance(initial_parameters, ParameterVector):
            raise TypeError("initial_parameters 必须是 ParameterVector")
        require_identifier(run_id, "run_id")
        require_identifier(dataset_id, "dataset_id")
        self.path = path
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.config_sha256 = training_config_sha256(config)
        self.initial_parameters = initial_parameters
        self.cohort_size = config.optimization.cohort_size
        self.rung_budgets = tuple(
            rung.budget_steps for rung in config.optimization.asha_rungs
        )
        self.rung_count = len(config.optimization.asha_rungs)

    def empty_state(self) -> ProductionScheduleState:
        """返回与当前 run/data/config/rung 身份绑定的初始空快照。"""

        return ProductionScheduleState(
            run_id=self.run_id,
            dataset_id=self.dataset_id,
            config_sha256=self.config_sha256,
            initial_parameters=self.initial_parameters,
            rung_count=self.rung_count,
        )

    def load(self) -> ProductionScheduleState:
        """不存在时返回空状态；存在时严格校验摘要、schema 和全部身份。"""

        if not self.path.exists():
            return self.empty_state()
        root = read_json_object(self.path)
        if set(root) != _ROOT_KEYS:
            raise SchemaMismatchError("生产调度状态根字段集合不匹配")
        if _integer(root, "schema_version") != PRODUCTION_SCHEDULE_SCHEMA_VERSION:
            raise SchemaMismatchError(
                f"生产调度状态仅支持 schema {PRODUCTION_SCHEDULE_SCHEMA_VERSION}"
            )
        payload = root["payload"]
        if not isinstance(payload, dict):
            raise SchemaMismatchError("生产调度 payload 必须是 JSON object")
        expected_sha = _string(root, "payload_sha256")
        try:
            require_sha256(expected_sha)
        except (TypeError, ValueError) as exc:
            raise SchemaMismatchError("payload_sha256 不是 SHA-256") from exc
        actual_sha = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        if actual_sha != expected_sha:
            raise IntegrityError("生产调度完整 payload SHA-256 不匹配")
        state = _state_from_json(payload)
        self._require_identity(state)
        return state

    def persist(self, state: ProductionScheduleState) -> ProductionScheduleState:
        """更新时间后原子提交完整快照，并返回磁盘中实际保存的状态。"""

        if not isinstance(state, ProductionScheduleState):
            raise TypeError("state 必须是 ProductionScheduleState")
        self._require_identity(state)
        committed = replace(state, updated_at_ms=time.time_ns() / 1_000_000.0)
        payload = _state_to_json(committed)
        atomic_write_json(
            self.path,
            {
                "schema_version": PRODUCTION_SCHEDULE_SCHEMA_VERSION,
                "payload": payload,
                "payload_sha256": hashlib.sha256(
                    _canonical_json_bytes(payload)
                ).hexdigest(),
            },
        )
        return committed

    def _require_identity(self, state: ProductionScheduleState) -> None:
        expected = (
            self.run_id,
            self.dataset_id,
            self.config_sha256,
            self.initial_parameters,
            self.rung_count,
        )
        actual = (
            state.run_id,
            state.dataset_id,
            state.config_sha256,
            state.initial_parameters,
            state.rung_count,
        )
        if actual != expected:
            raise SchemaMismatchError(
                "生产调度恢复状态与当前 run/data/config/initial/rungs 不一致"
            )
        for context in state.contexts:
            if context.cohort_index != context.trial_index // self.cohort_size:
                raise SchemaMismatchError(
                    "context.cohort_index 与配置 cohort_size/trial_index 不一致"
                )
            if context.budget_steps != self.rung_budgets[context.rung_index]:
                raise SchemaMismatchError(
                    "context.budget_steps 与配置 ASHA rung 累计预算不一致"
                )


def _validate_contexts(
    state: ProductionScheduleState,
) -> dict[tuple[int, int, CurriculumStage, int], ProductionTrialContext]:
    keys = tuple(context.key for context in state.contexts)
    if len(keys) != len(set(keys)):
        raise ValueError("contexts 不得包含重复 cohort/trial/stage/rung")
    if any(context.rung_index >= state.rung_count for context in state.contexts):
        raise ValueError("context.rung_index 超出配置 rung 数量")

    first_trial_order: list[int] = []
    first_cohort_order: list[int] = []
    parameters_by_trial: dict[int, ParameterVector] = {}
    cohort_by_trial: dict[int, int] = {}
    contexts_by_trial: dict[int, list[ProductionTrialContext]] = {}
    for context in state.contexts:
        if context.trial_index not in parameters_by_trial:
            first_trial_order.append(context.trial_index)
            first_cohort_order.append(context.cohort_index)
            parameters_by_trial[context.trial_index] = context.parameters
            cohort_by_trial[context.trial_index] = context.cohort_index
        elif parameters_by_trial[context.trial_index] != context.parameters:
            raise ValueError("同一 trial_index 的 parameters 不得变化")
        if cohort_by_trial[context.trial_index] != context.cohort_index:
            raise ValueError("同一 trial_index 不得跨 cohort")
        contexts_by_trial.setdefault(context.trial_index, []).append(context)
    if tuple(first_trial_order) != tuple(range(len(first_trial_order))):
        raise ValueError("proposal trial_index 必须按首次出现从 0 连续递增")
    if first_cohort_order:
        if first_cohort_order[0] != 0 or any(
            current not in (previous, previous + 1)
            for previous, current in zip(
                first_cohort_order,
                first_cohort_order[1:],
                strict=False,
            )
        ):
            raise ValueError("proposal cohort_index 必须从 0 开始且不得回退或跳号")
    if len(parameters_by_trial) != len(set(parameters_by_trial.values())):
        raise ValueError("不同 trial_index 不得重复 ParameterVector proposal")
    if state.contexts and parameters_by_trial[0] != state.initial_parameters:
        raise ValueError("trial 0 必须使用 initial_parameters")

    stage_indices = {stage: index for index, stage in enumerate(CURRICULUM_ORDER)}
    for trial_index, contexts in contexts_by_trial.items():
        first = contexts[0]
        if (
            first.curriculum_stage is not CurriculumStage.BASIC
            or first.rung_index != 0
            or first.parent_checkpoint_path is not None
        ):
            raise ValueError(
                f"trial {trial_index} 必须从 BASIC rung 0 且无 parent 开始"
            )
        for previous, current in zip(contexts, contexts[1:], strict=False):
            previous_stage = stage_indices[previous.curriculum_stage]
            current_stage = stage_indices[current.curriculum_stage]
            same_stage = current_stage == previous_stage
            next_stage = current_stage == previous_stage + 1
            if same_stage:
                if current.rung_index != previous.rung_index + 1:
                    raise ValueError("同一课程阶段的 rung 必须逐一递增")
                if current.budget_steps <= previous.budget_steps:
                    raise ValueError("同一课程阶段的累计 budget_steps 必须递增")
            elif not (
                next_stage
                and previous.rung_index == state.rung_count - 1
                and current.rung_index == 0
            ):
                raise ValueError("curriculum stage/rung 不得回退、跳级或跳 rung")
    return {context.key: context for context in state.contexts}


def _validate_jobs(
    state: ProductionScheduleState,
    contexts: dict[tuple[int, int, CurriculumStage, int], ProductionTrialContext],
) -> dict[tuple[int, int, CurriculumStage, int], ProductionJobRecord]:
    keys = tuple(job.context.key for job in state.jobs)
    if len(keys) != len(set(keys)):
        raise ValueError("jobs 不得重复 context key")
    if any(key not in contexts for key in keys):
        raise ValueError("job 必须引用已注册 context")
    if keys != tuple(context.key for context in state.contexts[: len(keys)]):
        raise ValueError("jobs 必须是 contexts 的 canonical 完成前缀")
    for job in state.jobs:
        if job.context != contexts[job.context.key]:
            raise ValueError("job.context 必须与注册 context 完全一致")
        if job.action is AshaAction.PROMOTE and (
            job.context.rung_index >= state.rung_count - 1
        ):
            raise ValueError("末级 rung 不得使用 PROMOTE")
        if job.action is AshaAction.CONTINUE and (
            job.context.rung_index != state.rung_count - 1
        ):
            raise ValueError("只有末级 rung 可以使用 CONTINUE")
        schedule_passed = (
            job.context.curriculum_stage is CurriculumStage.FULL
            and job.context.rung_index == state.rung_count - 1
            and job.action is AshaAction.CONTINUE
        )
        if job.acceptance.schedule is not schedule_passed:
            raise ValueError("schedule gate 只允许 FULL 末级 CONTINUE job 通过")
    return {job.context.key: job for job in state.jobs}


def _validate_transitions(
    state: ProductionScheduleState,
    jobs: dict[tuple[int, int, CurriculumStage, int], ProductionJobRecord],
) -> None:
    contexts_by_trial: dict[int, list[ProductionTrialContext]] = {}
    for context in state.contexts:
        contexts_by_trial.setdefault(context.trial_index, []).append(context)
    for contexts in contexts_by_trial.values():
        for previous, current in zip(contexts, contexts[1:], strict=False):
            job = jobs.get(previous.key)
            if job is None or job.action is None:
                raise ValueError("未完成或未决定 action 的 job 后不得注册下一 context")
            expected_action = (
                AshaAction.PROMOTE
                if previous.curriculum_stage is current.curriculum_stage
                else AshaAction.CONTINUE
            )
            if job.action is not expected_action:
                raise ValueError("PRUNE 或错误 ASHA action 后不得继续同一 trial")
            if current.parent_checkpoint_path != job.checkpoint_path:
                raise ValueError("下一 context 必须精确引用上一 job checkpoint")


def _validate_history(
    state: ProductionScheduleState,
    jobs: dict[tuple[int, int, CurriculumStage, int], ProductionJobRecord],
) -> None:
    trial_indices = tuple(observation.trial_index for observation in state.history)
    if trial_indices != tuple(range(len(trial_indices))):
        raise ValueError("history trial_index 必须从 0 连续递增")
    contexts_by_trial: dict[int, list[ProductionTrialContext]] = {}
    for context in state.contexts:
        contexts_by_trial.setdefault(context.trial_index, []).append(context)
    passed_count = 0
    for observation in state.history:
        contexts = contexts_by_trial.get(observation.trial_index)
        if not contexts:
            raise ValueError("history 不得引用未注册 proposal")
        latest = contexts[-1]
        if observation.parameters != latest.parameters:
            raise ValueError("history parameters 必须与 proposal 一致")
        job = jobs.get(latest.key)
        if job is None or job.action is None:
            raise ValueError("history 只能引用已有最终 action 的 job")
        if observation.objective != job.objective:
            raise ValueError("history objective 必须来自 trial 最后一个 job")
        if observation.acceptance != job.acceptance:
            raise ValueError("history acceptance 必须来自 trial 最后一个 job")
        if observation.acceptance.passed:
            passed_count += 1
            if not (
                latest.curriculum_stage is CurriculumStage.FULL
                and latest.rung_index == state.rung_count - 1
                and job.action is AshaAction.CONTINUE
            ):
                raise ValueError("PASSED history 必须来自 FULL 末级 CONTINUE")
        elif job.action is not AshaAction.PRUNE:
            raise ValueError("未通过 history 必须由 PRUNE 终结")
    if passed_count > 1:
        raise ValueError("生产调度 history 最多只能包含一个通过 winner")


def _domain_gates_passed(acceptance: TrialAcceptance) -> bool:
    """调度 gate 独立于领域 gate；ASHA rank prune 不抹掉领域通过事实。"""

    names = tuple(field.name for field in fields(TrialAcceptance))
    domain_names = tuple(name for name in names if name != "schedule")
    return all(getattr(acceptance, name) for name in domain_names)


def _artifact_pair(
    name: str,
    *,
    path: Path | None,
    sha256: str | None,
) -> None:
    _optional_path(f"{name}_path", path)
    if (path is None) != (sha256 is None):
        raise ValueError(f"{name}_path 与 {name}_sha256 必须同时存在或同时为空")
    if sha256 is not None:
        require_sha256(sha256)


def _optional_path(name: str, value: Path | None) -> None:
    if value is None:
        return
    if not isinstance(value, Path):
        raise TypeError(f"{name} 必须是 pathlib.Path 或 None")
    rendered = str(value)
    if not rendered or "\x00" in rendered:
        raise ValueError(f"{name} 必须是有效的非空路径")


def _state_to_json(state: ProductionScheduleState) -> dict[str, object]:
    return {
        "run_id": state.run_id,
        "dataset_id": state.dataset_id,
        "config_sha256": state.config_sha256,
        "initial_parameters": _parameters_to_json(state.initial_parameters),
        "rung_count": state.rung_count,
        "contexts": [_context_to_json(context) for context in state.contexts],
        "jobs": [_job_to_json(job) for job in state.jobs],
        "history": [_observation_to_json(observation) for observation in state.history],
        "updated_at_ms": float(state.updated_at_ms),
    }


def _state_from_json(payload: dict[str, object]) -> ProductionScheduleState:
    if set(payload) != _PAYLOAD_KEYS:
        raise SchemaMismatchError("生产调度 payload 字段集合不匹配")
    contexts = _array(payload, "contexts")
    jobs = _array(payload, "jobs")
    history = _array(payload, "history")
    initial_parameters = payload["initial_parameters"]
    if not isinstance(initial_parameters, dict):
        raise SchemaMismatchError("initial_parameters 必须是 JSON object")
    try:
        return ProductionScheduleState(
            run_id=_string(payload, "run_id"),
            dataset_id=_string(payload, "dataset_id"),
            config_sha256=_string(payload, "config_sha256"),
            initial_parameters=_parameters_from_json(initial_parameters),
            rung_count=_integer(payload, "rung_count"),
            contexts=tuple(_context_from_json(item) for item in contexts),
            jobs=tuple(_job_from_json(item) for item in jobs),
            history=tuple(_observation_from_json(item) for item in history),
            updated_at_ms=_number(payload, "updated_at_ms"),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError("生产调度 typed schema 不匹配") from exc


def _context_to_json(context: ProductionTrialContext) -> dict[str, object]:
    return {
        "cohort_index": context.cohort_index,
        "trial_index": context.trial_index,
        "parameters": _parameters_to_json(context.parameters),
        "curriculum_stage": context.curriculum_stage.value,
        "rung_index": context.rung_index,
        "budget_steps": context.budget_steps,
        "parent_checkpoint_path": _path_to_json(context.parent_checkpoint_path),
    }


def _context_from_json(payload: object) -> ProductionTrialContext:
    raw = _object(payload, "context")
    if set(raw) != _CONTEXT_KEYS:
        raise SchemaMismatchError("context 字段集合不匹配")
    parameters = raw["parameters"]
    if not isinstance(parameters, dict):
        raise SchemaMismatchError("context.parameters 必须是 JSON object")
    try:
        return ProductionTrialContext(
            cohort_index=_integer(raw, "cohort_index"),
            trial_index=_integer(raw, "trial_index"),
            parameters=_parameters_from_json(parameters),
            curriculum_stage=CurriculumStage(_string(raw, "curriculum_stage")),
            rung_index=_integer(raw, "rung_index"),
            budget_steps=_integer(raw, "budget_steps"),
            parent_checkpoint_path=_path_from_json(raw, "parent_checkpoint_path"),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError("context typed schema 不匹配") from exc


def _job_to_json(job: ProductionJobRecord) -> dict[str, object]:
    return {
        "context": _context_to_json(job.context),
        "objective": float(job.objective),
        "acceptance": _acceptance_to_json(job.acceptance),
        "gate_passed": job.gate_passed,
        "action": None if job.action is None else job.action.value,
        "checkpoint_path": _path_to_json(job.checkpoint_path),
        "checkpoint_sha256": job.checkpoint_sha256,
        "feedback_path": _path_to_json(job.feedback_path),
        "feedback_sha256": job.feedback_sha256,
    }


def _job_from_json(payload: object) -> ProductionJobRecord:
    raw = _object(payload, "job")
    if set(raw) != _JOB_KEYS:
        raise SchemaMismatchError("job 字段集合不匹配")
    acceptance = raw["acceptance"]
    if not isinstance(acceptance, dict):
        raise SchemaMismatchError("job.acceptance 必须是 JSON object")
    action = raw["action"]
    if action is not None and not isinstance(action, str):
        raise SchemaMismatchError("job.action 必须是字符串或 null")
    try:
        return ProductionJobRecord(
            context=_context_from_json(raw["context"]),
            objective=_number(raw, "objective"),
            acceptance=_acceptance_from_json(acceptance),
            gate_passed=_boolean(raw, "gate_passed"),
            action=None if action is None else AshaAction(action),
            checkpoint_path=_path_from_json(raw, "checkpoint_path"),
            checkpoint_sha256=_optional_string(raw, "checkpoint_sha256"),
            feedback_path=_path_from_json(raw, "feedback_path"),
            feedback_sha256=_optional_string(raw, "feedback_sha256"),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError("job typed schema 不匹配") from exc


def _observation_to_json(observation: TrialObservation) -> dict[str, object]:
    return {
        "trial_index": observation.trial_index,
        "parameters": _parameters_to_json(observation.parameters),
        "objective": float(observation.objective),
        "acceptance": _acceptance_to_json(observation.acceptance),
    }


def _observation_from_json(payload: object) -> TrialObservation:
    raw = _object(payload, "history item")
    if set(raw) != _OBSERVATION_KEYS:
        raise SchemaMismatchError("history item 字段集合不匹配")
    parameters = raw["parameters"]
    acceptance = raw["acceptance"]
    if not isinstance(parameters, dict) or not isinstance(acceptance, dict):
        raise SchemaMismatchError("history parameters/acceptance 必须是 object")
    try:
        return TrialObservation(
            trial_index=_integer(raw, "trial_index"),
            parameters=_parameters_from_json(parameters),
            objective=_number(raw, "objective"),
            acceptance=_acceptance_from_json(acceptance),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError("history item typed schema 不匹配") from exc


def _parameters_to_json(parameters: ParameterVector) -> dict[str, object]:
    return {name: getattr(parameters, name) for name in _PARAMETER_FIELDS}


def _parameters_from_json(payload: dict[str, object]) -> ParameterVector:
    if set(payload) != set(_PARAMETER_FIELDS):
        raise SchemaMismatchError("parameters 字段集合不匹配")
    values: list[float | int] = []
    for name in _PARAMETER_FIELDS:
        if name == "max_candidates":
            values.append(_integer(payload, name))
        else:
            values.append(_number(payload, name))
    return ParameterVector(*values)  # type: ignore[arg-type]


def _acceptance_to_json(acceptance: TrialAcceptance) -> dict[str, object]:
    return {
        field.name: getattr(acceptance, field.name) for field in fields(TrialAcceptance)
    }


def _acceptance_from_json(payload: dict[str, object]) -> TrialAcceptance:
    names = tuple(field.name for field in fields(TrialAcceptance))
    if set(payload) != set(names):
        raise SchemaMismatchError("acceptance 字段集合不匹配")
    return TrialAcceptance(*(_boolean(payload, name) for name in names))


def _path_to_json(value: Path | None) -> str | None:
    return None if value is None else str(value)


def _path_from_json(payload: dict[str, object], key: str) -> Path | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SchemaMismatchError(f"{key} 必须是有效路径字符串或 null")
    return Path(value)


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SchemaMismatchError("无法计算生产调度 canonical JSON") from exc


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SchemaMismatchError(f"{name} 必须是 JSON object")
    return value


def _array(payload: dict[str, object], key: str) -> list[object]:
    value = payload[key]
    if not isinstance(value, list):
        raise SchemaMismatchError(f"{key} 必须是 JSON array")
    return value


def _string(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise SchemaMismatchError(f"{key} 必须是字符串")
    return value


def _optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload[key]
    if value is not None and not isinstance(value, str):
        raise SchemaMismatchError(f"{key} 必须是字符串或 null")
    return value


def _integer(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaMismatchError(f"{key} 必须是整数")
    return value


def _number(payload: dict[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaMismatchError(f"{key} 必须是数值")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaMismatchError(f"{key} 必须是有限数值")
    return result


def _boolean(payload: dict[str, object], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise SchemaMismatchError(f"{key} 必须是 bool")
    return value


__all__ = (
    "PRODUCTION_SCHEDULE_SCHEMA_VERSION",
    "ProductionJobRecord",
    "ProductionScheduleState",
    "ProductionScheduleStore",
    "ProductionTrialContext",
)
