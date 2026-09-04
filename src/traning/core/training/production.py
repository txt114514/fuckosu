"""真实数据、持续参数搜索、断点恢复与 checkpoint 发布的生产入口。"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
from pathlib import Path
import time
import uuid

from package import CurriculumStage

from traning.conf import V2Config
from traning.state import TelemetryEvent
from traning.core.data import TrainingDatasetBundle, require_quality
from traning.lib.infrastructure import IntegrityError, sha256_file
from traning.lib.telemetry import StateStore, TelemetryReporter
from traning.core.training.checkpoints import load_runtime_checkpoint
from traning.core.training.evaluator import acceptance_from_orchestration
from traning.core.training.optimization import (
    PARAMETER_REGISTRY,
    DeterministicSearchController,
    ParameterVector,
    SearchDecision,
    SearchExhaustedError,
    SearchStatus,
    TrialAcceptance,
    TrialObservation,
)
from traning.core.training.orchestration import TrainingOrchestrator

from .production_contracts import (
    ProductionGateSpec,
    ProductionTrainingResult,
)
from .production_stages import (
    ProductionStageRunner,
    config_for_parameters,
)
from .hard_example_feedback import (
    HardExampleFeedbackArtifact,
    HardExampleFeedbackStore,
)
from .production_schedule import (
    ProductionJobRecord,
    ProductionScheduleState,
    ProductionScheduleStore,
    ProductionTrialContext,
)
from .scheduling import (
    AshaAction,
    AshaRung,
    AshaScheduler,
    AshaTrial,
    CurriculumAction,
    CurriculumGate,
    decide_curriculum,
)


SEARCH_STATE_FILENAME = "search_state.json"
"""每个 run 唯一的原子搜索恢复文件名。"""


@dataclass(frozen=True, slots=True)
class ProductionTrainer:
    """把已检查的真实数据 bundle 接入可恢复的门禁驱动搜索。"""

    config: V2Config
    datasets: TrainingDatasetBundle
    gates: ProductionGateSpec = ProductionGateSpec()

    def __post_init__(self) -> None:
        if not isinstance(self.config, V2Config):
            raise TypeError("config 必须是 V2Config")
        if not isinstance(self.datasets, TrainingDatasetBundle):
            raise TypeError("datasets 必须是 TrainingDatasetBundle")
        if not isinstance(self.gates, ProductionGateSpec):
            raise TypeError("gates 必须是 ProductionGateSpec")

    def run(
        self,
        *,
        run_dir: Path,
        run_id: str,
        resume: bool = True,
        reporter: TelemetryReporter | None = None,
    ) -> ProductionTrainingResult:
        """持续尝试未重复参数，只有全部门禁通过才发布并返回模型。"""

        if not isinstance(run_dir, Path):
            raise TypeError("run_dir 必须是 pathlib.Path")
        if not isinstance(run_id, str) or not run_id or run_id != run_id.strip():
            raise ValueError("run_id 必须非空且无首尾空格")
        if not isinstance(resume, bool):
            raise TypeError("resume 必须是 bool")
        require_quality(self.datasets.quality_report)
        if (
            self.datasets.coordinate_transform is None
        ):  # pragma: no cover - quality 已阻断
            raise RuntimeError("数据 bundle 缺少坐标变换")

        run_dir.mkdir(parents=True, exist_ok=True)
        selected_reporter = reporter or TelemetryReporter(
            run_id,
            StateStore(
                run_dir / "telemetry",
                schema_version=self.config.telemetry.schema_version,
            ),
        )
        if selected_reporter.run_id != run_id:
            raise ValueError("reporter.run_id 与生产 run_id 不一致")

        initial = _initial_parameter_vector(self.config)
        schedule_store = ProductionScheduleStore(
            run_dir / SEARCH_STATE_FILENAME,
            run_id=run_id,
            dataset_id=self.datasets.dataset_identity,
            config=self.config,
            initial_parameters=initial,
        )
        if not resume and schedule_store.path.exists():
            raise FileExistsError("搜索状态已存在；请启用 resume 或使用新的 run_id")
        state = schedule_store.load() if resume else schedule_store.empty_state()
        runners: dict[tuple[int, int, CurriculumStage, int], ProductionStageRunner] = {}
        controller = DeterministicSearchController(
            self.config.training.seed,
            self.config.optimization.max_trials,
        )
        asha = AshaScheduler(
            tuple(
                AshaRung(
                    index=index,
                    budget=rung.budget_steps,
                    promotion_fraction=rung.promotion_fraction,
                )
                for index, rung in enumerate(self.config.optimization.asha_rungs)
            )
        )
        try:
            _validate_job_artifacts(state)
            while True:
                active_cohort = _active_cohort_index(state)
                if active_cohort is not None:
                    previous_history_size = len(state.history)
                    state = _complete_active_cohort(
                        state,
                        cohort_index=active_cohort,
                        trainer=self,
                        run_dir=run_dir,
                        run_id=run_id,
                        reporter=selected_reporter,
                        store=schedule_store,
                        asha=asha,
                        runners=runners,
                    )
                    for observation in state.history[previous_history_size:]:
                        _publish_trial_completed(selected_reporter, observation)
                    continue

                decision = controller.decide(initial, state.history)
                if decision.status is SearchStatus.PASSED:
                    winner = decision.best_observation
                    if winner is None:  # pragma: no cover - DTO 已保证
                        raise RuntimeError("PASSED 缺少 winning observation")
                    break
                if decision.status is SearchStatus.EXHAUSTED:
                    raise SearchExhaustedError(decision)
                state = _start_cohort(
                    state,
                    controller=controller,
                    initial=initial,
                    config=self.config,
                    store=schedule_store,
                )
        except SearchExhaustedError as error:
            _publish_search_terminal(
                selected_reporter,
                event_type="search.exhausted",
                observation=error.decision.best_observation,
                trial_count=error.decision.trial_count,
            )
            raise
        except Exception as error:
            _publish_search_failure(selected_reporter, error)
            raise
        try:
            winning_job = _latest_job_for_trial(state, winner.trial_index)
            checkpoint_directory = winning_job.checkpoint_path
            if checkpoint_directory is None:  # pragma: no cover - state 已保证
                raise RuntimeError("winning job 缺少 checkpoint artifact")
            winning_config = config_for_parameters(self.config, winner.parameters)
            # 即使 winner 来自恢复历史，也必须重新执行完整摘要/config/坐标校验。
            load_runtime_checkpoint(
                checkpoint_directory,
                winning_config,
                self.datasets.coordinate_transform,
                expected_dataset_id=self.datasets.dataset_identity,
            )
        except Exception as error:
            _publish_search_failure(selected_reporter, error)
            raise
        _publish_search_terminal(
            selected_reporter,
            event_type="search.passed",
            observation=winner,
            trial_count=len(state.history),
        )
        winning_runner = runners.get(winning_job.context.key)
        return ProductionTrainingResult(
            observation=winner,
            winning_config=winning_config,
            checkpoint_directory=checkpoint_directory,
            stage_results=(
                () if winning_runner is None else tuple(winning_runner.stage_results)
            ),
            metrics=None if winning_runner is None else winning_runner.metrics,
            resumed=winning_runner is None,
        )


def _initial_parameter_vector(config: V2Config) -> ParameterVector:
    """从唯一配置构造搜索空间的初始 proposal。"""

    return PARAMETER_REGISTRY.normalize(
        ParameterVector(
            learning_rate=config.training.learning_rate,
            score_threshold=config.perception.score_threshold,
            max_candidates=config.perception.max_candidates,
            risk_lambda=config.decision.risk_lambda,
            wait_cost=config.decision.wait_cost,
            min_confidence=config.decision.min_confidence,
        )
    )


def _start_cohort(
    state: ProductionScheduleState,
    *,
    controller: DeterministicSearchController,
    initial: ParameterVector,
    config: V2Config,
    store: ProductionScheduleStore,
) -> ProductionScheduleState:
    """从未尝试参数中同步填充一个新 cohort，并先提交 context ledger。"""

    contexts = list(state.contexts)
    cohort_index = state.next_trial_index // config.optimization.cohort_size
    added = 0
    terminal: SearchDecision | None = None
    while added < config.optimization.cohort_size:
        ledger = _proposal_ledger(replace(state, contexts=tuple(contexts)))
        decision = controller.decide(initial, ledger)
        if decision.status is not SearchStatus.RUNNING:
            terminal = decision
            break
        proposal = decision.proposal
        if proposal is None:  # pragma: no cover - DTO 已保证
            raise RuntimeError("RUNNING 缺少 proposal")
        contexts.append(
            ProductionTrialContext(
                cohort_index=cohort_index,
                trial_index=len(ledger),
                parameters=proposal,
                curriculum_stage=CurriculumStage.BASIC,
                rung_index=0,
                budget_steps=config.optimization.asha_rungs[0].budget_steps,
            )
        )
        added += 1
    if added == 0:
        if terminal is None:  # pragma: no cover - 循环至少执行一次
            raise RuntimeError("无法建立新 cohort")
        if terminal.status is SearchStatus.EXHAUSTED:
            raise SearchExhaustedError(terminal)
        raise RuntimeError("已有 winner 时不得建立新 cohort")
    return store.persist(replace(state, contexts=tuple(contexts)))


def _proposal_ledger(
    state: ProductionScheduleState,
) -> tuple[TrialObservation, ...]:
    """把 active proposal 投影为未通过占位观测，仅用于避免重复提案。"""

    completed = {item.trial_index: item for item in state.history}
    parameters = state.tried_parameters
    placeholder = TrialAcceptance(
        data=False,
        perception=False,
        tracking=False,
        belief=False,
        outcome=False,
        decision=False,
        golden=False,
        schedule=False,
    )
    return tuple(
        completed.get(
            trial_index,
            TrialObservation(
                trial_index=trial_index,
                parameters=parameter,
                objective=0.0,
                acceptance=placeholder,
            ),
        )
        for trial_index, parameter in enumerate(parameters)
    )


def _active_cohort_index(state: ProductionScheduleState) -> int | None:
    """返回唯一尚未写入 final history 的 cohort。"""

    finalized = {item.trial_index for item in state.history}
    active = {
        context.cohort_index
        for context in state.contexts
        if context.trial_index not in finalized
    }
    if len(active) > 1:
        raise RuntimeError("调度状态不得同时存在多个 active cohort")
    return None if not active else next(iter(active))


def _complete_active_cohort(
    state: ProductionScheduleState,
    *,
    cohort_index: int,
    trainer: ProductionTrainer,
    run_dir: Path,
    run_id: str,
    reporter: TelemetryReporter,
    store: ProductionScheduleStore,
    asha: AshaScheduler,
    runners: dict[tuple[int, int, CurriculumStage, int], ProductionStageRunner],
) -> ProductionScheduleState:
    """恢复或推进一个 cohort，直到全部 prune 或产生 FULL winner。"""

    while True:
        if len(state.jobs) < len(state.contexts):
            context = state.contexts[len(state.jobs)]
            if context.cohort_index != cohort_index:
                raise RuntimeError("pending context 不属于 active cohort")
            state = _execute_job(
                state,
                context=context,
                trainer=trainer,
                run_dir=run_dir,
                run_id=run_id,
                reporter=reporter,
                store=store,
                runners=runners,
            )
            continue

        undecided = tuple(
            job
            for job in state.jobs
            if job.context.cohort_index == cohort_index and job.action is None
        )
        if undecided:
            state = _decide_jobs(state, undecided=undecided, asha=asha, store=store)
            continue

        successors = _successor_contexts(
            state,
            cohort_index=cohort_index,
            config=trainer.config,
        )
        if successors:
            state = store.persist(
                replace(state, contexts=(*state.contexts, *successors))
            )
            continue
        return _finalize_cohort(state, cohort_index=cohort_index, store=store)


def _execute_job(
    state: ProductionScheduleState,
    *,
    context: ProductionTrialContext,
    trainer: ProductionTrainer,
    run_dir: Path,
    run_id: str,
    reporter: TelemetryReporter,
    store: ProductionScheduleStore,
    runners: dict[tuple[int, int, CurriculumStage, int], ProductionStageRunner],
) -> ProductionScheduleState:
    """执行一个真实资源 job，并在状态前先提交 checkpoint 与 feedback。"""

    input_feedback = _input_feedback_for_context(
        state,
        context=context,
        trainer=trainer,
        run_id=run_id,
        config_sha256=store.config_sha256,
    )
    runner = ProductionStageRunner(
        base_config=trainer.config,
        context=context,
        datasets=trainer.datasets,
        gates=trainer.gates,
        run_dir=run_dir,
        run_id=run_id,
        reporter=reporter,
        input_feedback=input_feedback,
    )
    runners[context.key] = runner
    orchestration = TrainingOrchestrator(runner).run(trainer.datasets.quality_report)
    acceptance = replace(
        acceptance_from_orchestration(orchestration),
        schedule=False,
    )
    gate_passed = replace(acceptance, schedule=True).passed

    job_directory = _job_directory(run_dir, context)
    checkpoint_directory = job_directory / f"checkpoint-{uuid.uuid4().hex}"
    runner.publish_job_checkpoint(checkpoint_directory)
    checkpoint_sha256 = _directory_sha256(checkpoint_directory)

    feedback_path = job_directory / "hard-example-feedback.json"
    feedback_store = _feedback_store(
        feedback_path,
        trainer=trainer,
        run_id=run_id,
        config_sha256=store.config_sha256,
    )
    feedback_store.persist(
        runner.hard_example_plan,
        source_trial_index=context.trial_index,
        source_parameters=context.parameters,
        evaluated=runner.feedback_evaluated,
        bonus=trainer.config.optimization.hard_example_bonus,
        max_weight=trainer.config.optimization.hard_example_max_weight,
    )
    feedback_sha256 = sha256_file(feedback_path)
    job = ProductionJobRecord(
        context=context,
        objective=runner.metrics.objective,
        acceptance=acceptance,
        gate_passed=gate_passed,
        checkpoint_path=checkpoint_directory,
        checkpoint_sha256=checkpoint_sha256,
        feedback_path=feedback_path,
        feedback_sha256=feedback_sha256,
    )
    committed = store.persist(replace(state, jobs=(*state.jobs, job)))
    _publish_job_completed(reporter, job)
    return committed


def _feedback_store(
    path: Path,
    *,
    trainer: ProductionTrainer,
    run_id: str,
    config_sha256: str,
) -> HardExampleFeedbackStore:
    """构造始终绑定完整 TRAIN split 的 feedback store。"""

    transform = trainer.datasets.transform_fingerprint
    if transform is None:  # pragma: no cover - quality 门已阻断
        raise RuntimeError("feedback store 缺少 transform fingerprint")
    return HardExampleFeedbackStore(
        path,
        run_id=run_id,
        dataset_id=trainer.datasets.dataset_identity,
        config_sha256=config_sha256,
        transform_fingerprint=transform,
        train_dataset=trainer.datasets.train,
    )


def _input_feedback_for_context(
    state: ProductionScheduleState,
    *,
    context: ProductionTrialContext,
    trainer: ProductionTrainer,
    run_id: str,
    config_sha256: str,
) -> HardExampleFeedbackArtifact | None:
    """优先加载直接 parent；新 cohort 只读上一 cohort 已提交的最终反馈。"""

    source: ProductionJobRecord | None = None
    if context.parent_checkpoint_path is not None:
        source = next(
            (
                job
                for job in state.jobs
                if job.checkpoint_path == context.parent_checkpoint_path
            ),
            None,
        )
        if source is None:
            raise IntegrityError("parent checkpoint 没有对应已提交 job")
    elif context.cohort_index > 0:
        previous_trials = tuple(
            item.trial_index
            for item in state.history
            if item.trial_index < context.trial_index
        )
        if previous_trials:
            source = _latest_job_for_trial(state, max(previous_trials))
    if source is None:
        return None
    if source.feedback_path is None or source.feedback_sha256 is None:
        raise IntegrityError("source job 缺少 feedback artifact")
    if sha256_file(source.feedback_path) != source.feedback_sha256:
        raise IntegrityError("hard-example feedback 文件 SHA-256 不匹配")
    feedback_store = _feedback_store(
        source.feedback_path,
        trainer=trainer,
        run_id=run_id,
        config_sha256=config_sha256,
    )
    return feedback_store.load(
        expected_source_trial_index=source.context.trial_index,
        expected_source_parameters=source.context.parameters,
    )


def _decide_jobs(
    state: ProductionScheduleState,
    *,
    undecided: tuple[ProductionJobRecord, ...],
    asha: AshaScheduler,
    store: ProductionScheduleStore,
) -> ProductionScheduleState:
    """同一 cohort/stage/rung 完成后一次性执行稳定 ASHA 决策。"""

    frontier = {
        (job.context.cohort_index, job.context.curriculum_stage, job.context.rung_index)
        for job in undecided
    }
    if len(frontier) != 1:
        raise RuntimeError("ASHA 只能比较同 cohort/stage/rung 的 completed jobs")
    sample = undecided[0].context
    decisions = asha.decide(
        sample.rung_index,
        tuple(
            AshaTrial(
                trial_id=str(job.context.trial_index),
                rung_index=job.context.rung_index,
                objective=job.objective,
                gate_passed=job.gate_passed,
            )
            for job in undecided
        ),
    )
    actions = {int(item.trial_id): item.action for item in decisions}
    if (
        sample.curriculum_stage is CurriculumStage.FULL
        and sample.rung_index == state.rung_count - 1
    ):
        continuers = sorted(
            (
                job
                for job in undecided
                if actions[job.context.trial_index] is AshaAction.CONTINUE
            ),
            key=lambda job: (-job.objective, job.context.trial_index),
        )
        for loser in continuers[1:]:
            actions[loser.context.trial_index] = AshaAction.PRUNE

    undecided_keys = {job.context.key for job in undecided}
    jobs = []
    for job in state.jobs:
        if job.context.key not in undecided_keys:
            jobs.append(job)
            continue
        action = actions[job.context.trial_index]
        schedule_passed = (
            action is AshaAction.CONTINUE
            and job.context.curriculum_stage is CurriculumStage.FULL
            and job.context.rung_index == state.rung_count - 1
        )
        jobs.append(
            replace(
                job,
                action=action,
                acceptance=replace(job.acceptance, schedule=schedule_passed),
            )
        )
    return store.persist(replace(state, jobs=tuple(jobs)))


def _successor_contexts(
    state: ProductionScheduleState,
    *,
    cohort_index: int,
    config: V2Config,
) -> tuple[ProductionTrialContext, ...]:
    """依据已提交 action 创建同 proposal 的下一 rung 或下一课程 context。"""

    latest_contexts: dict[int, ProductionTrialContext] = {}
    for context in state.contexts:
        if context.cohort_index == cohort_index:
            latest_contexts[context.trial_index] = context
    jobs = {job.context.key: job for job in state.jobs}
    successors = []
    for trial_index in sorted(latest_contexts):
        context = latest_contexts[trial_index]
        job = jobs.get(context.key)
        if job is None or job.action is None:
            raise RuntimeError("无法从未完成 job 构造 successor")
        if job.action is AshaAction.PRUNE:
            continue
        if job.checkpoint_path is None:  # pragma: no cover - DTO 已保证
            raise RuntimeError("晋级 job 缺少 checkpoint")
        if job.action is AshaAction.PROMOTE:
            next_stage = context.curriculum_stage
            next_rung = context.rung_index + 1
        else:
            gates = tuple(
                CurriculumGate(field.name, getattr(job.acceptance, field.name))
                for field in fields(TrialAcceptance)
                if field.name != "schedule"
            )
            curriculum = decide_curriculum(context.curriculum_stage, gates)
            if curriculum.action is CurriculumAction.COMPLETE:
                continue
            if curriculum.action is not CurriculumAction.ADVANCE:
                raise RuntimeError("通过 terminal rung 后 curriculum 不得 HOLD")
            next_stage = curriculum.next_stage
            next_rung = 0
        successors.append(
            ProductionTrialContext(
                cohort_index=cohort_index,
                trial_index=trial_index,
                parameters=context.parameters,
                curriculum_stage=next_stage,
                rung_index=next_rung,
                # 下一 rung 尚未进入调度账本，预算必须来自唯一配置规格，
                # 不能从现有 context 反查一个本来就还不存在的值。
                budget_steps=config.optimization.asha_rungs[next_rung].budget_steps,
                parent_checkpoint_path=job.checkpoint_path,
            )
        )
    return tuple(successors)


def _finalize_cohort(
    state: ProductionScheduleState,
    *,
    cohort_index: int,
    store: ProductionScheduleStore,
) -> ProductionScheduleState:
    """按 trial index 将 cohort 终态一次性追加到连续 history。"""

    trial_indices = sorted(
        {
            context.trial_index
            for context in state.contexts
            if context.cohort_index == cohort_index
        }
    )
    observations = []
    for trial_index in trial_indices:
        job = _latest_job_for_trial(state, trial_index)
        if job.action not in (AshaAction.PRUNE, AshaAction.CONTINUE):
            raise RuntimeError("cohort 只能由 PRUNE 或 FULL CONTINUE 终结")
        observations.append(
            TrialObservation(
                trial_index=trial_index,
                parameters=job.context.parameters,
                objective=job.objective,
                acceptance=job.acceptance,
            )
        )
    return store.persist(replace(state, history=(*state.history, *observations)))


def _latest_job_for_trial(
    state: ProductionScheduleState,
    trial_index: int,
) -> ProductionJobRecord:
    """返回一个 proposal 按 context 顺序最后完成的 job。"""

    candidates = tuple(
        job for job in state.jobs if job.context.trial_index == trial_index
    )
    if not candidates:
        raise RuntimeError(f"trial {trial_index} 没有 completed job")
    return candidates[-1]


def _job_directory(run_dir: Path, context: ProductionTrialContext) -> Path:
    """返回不会让不同 curriculum/rung 互相覆盖的 job 目录。"""

    return (
        run_dir
        / "trials"
        / f"trial-{context.trial_index:06d}"
        / "jobs"
        / context.curriculum_stage.value
        / f"rung-{context.rung_index:02d}"
    )


def _directory_sha256(directory: Path) -> str:
    """摘要目录内所有相对路径与文件摘要，供 parent checkpoint 恢复校验。"""

    if not directory.is_dir():
        raise IntegrityError(f"checkpoint directory 不存在：{directory}")
    files_in_directory = tuple(
        sorted(path for path in directory.rglob("*") if path.is_file())
    )
    if not files_in_directory:
        raise IntegrityError("checkpoint directory 不得为空")
    digest = hashlib.sha256()
    for path in files_in_directory:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _validate_job_artifacts(state: ProductionScheduleState) -> None:
    """恢复前验证所有 checkpoint/feedback 外部内容与 state 摘要一致。"""

    for job in state.jobs:
        if job.checkpoint_path is None or job.checkpoint_sha256 is None:
            raise IntegrityError("completed job 缺少 checkpoint 摘要")
        if _directory_sha256(job.checkpoint_path) != job.checkpoint_sha256:
            raise IntegrityError("checkpoint directory SHA-256 不匹配")
        if job.feedback_path is None or job.feedback_sha256 is None:
            raise IntegrityError("completed job 缺少 feedback 摘要")
        if sha256_file(job.feedback_path) != job.feedback_sha256:
            raise IntegrityError("hard-example feedback 文件 SHA-256 不匹配")


def _publish_job_completed(
    reporter: TelemetryReporter,
    job: ProductionJobRecord,
) -> None:
    """发布一个已原子提交资源 job 的事实事件。"""

    reporter.publish(
        TelemetryEvent(
            schema_version=reporter.store.snapshot().schema_version,
            event_type="search.job.completed",
            timestamp_ms=time.time_ns() / 1_000_000.0,
            run_id=reporter.run_id,
            metrics=(("objective", job.objective),),
            payload=(
                ("budget_steps", job.context.budget_steps),
                ("curriculum_stage", job.context.curriculum_stage.value),
                ("rung_index", job.context.rung_index),
                ("trial_index", job.context.trial_index),
            ),
        )
    )


def _publish_trial_completed(
    reporter: TelemetryReporter,
    observation: TrialObservation,
) -> None:
    """发布一个唯一参数 proposal 已形成调度终态的事实事件。"""

    reporter.publish(
        TelemetryEvent(
            schema_version=reporter.store.snapshot().schema_version,
            event_type="search.trial.completed",
            timestamp_ms=time.time_ns() / 1_000_000.0,
            run_id=reporter.run_id,
            metrics=(("objective", observation.objective),),
            payload=(
                ("acceptance_passed", observation.acceptance.passed),
                ("trial_index", observation.trial_index),
            ),
        )
    )


def _publish_search_terminal(
    reporter: TelemetryReporter,
    *,
    event_type: str,
    observation: TrialObservation | None,
    trial_count: int,
) -> None:
    """发布通过或耗尽终态，不把普通门禁失败伪装成进程停止。"""

    metrics = () if observation is None else (("objective", observation.objective),)
    payload = (
        ("trial_count", trial_count),
        (
            "best_trial_index",
            None if observation is None else observation.trial_index,
        ),
    )
    reporter.publish(
        TelemetryEvent(
            schema_version=reporter.store.snapshot().schema_version,
            event_type=event_type,
            timestamp_ms=time.time_ns() / 1_000_000.0,
            run_id=reporter.run_id,
            metrics=metrics,
            payload=payload,
        )
    )


def _publish_search_failure(
    reporter: TelemetryReporter,
    error: Exception,
) -> None:
    """把不可恢复异常发布为明确 FAILED 终态后保持原异常传播。"""

    reporter.publish(
        TelemetryEvent(
            schema_version=reporter.store.snapshot().schema_version,
            event_type="search.failed",
            timestamp_ms=time.time_ns() / 1_000_000.0,
            run_id=reporter.run_id,
            payload=(
                ("error_type", type(error).__name__),
                ("message", str(error)),
            ),
        )
    )


__all__ = (
    "SEARCH_STATE_FILENAME",
    "ProductionTrainer",
    "ProductionGateSpec",
    "ProductionTrainingResult",
)
