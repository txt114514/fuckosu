from __future__ import annotations

import traceback
from typing import Callable, Generic, Iterable, Mapping, TypeVar

from loguru import logger
from prefect import flow

from before_traning.Lib.common.failures import format_exception
from before_traning.Lib.tasks.tasks import TaskRegistry, TaskSpec


SettingsT = TypeVar("SettingsT")
SettingsLoader = Callable[[], SettingsT]
ContinueOnError = Callable[[SettingsT], bool]


class TaskPipeline(Generic[SettingsT]):
    def __init__(
        self,
        registry: TaskRegistry[SettingsT],
        *,
        settings_loader: SettingsLoader[SettingsT],
        continue_on_error: ContinueOnError[SettingsT],
        flow_name: str,
    ):
        self.registry = registry
        self.settings_loader = settings_loader
        self.continue_on_error = continue_on_error
        self.flow_name = flow_name
        self._prefect_flow = flow(
            name=flow_name,
            log_prints=True,
        )(self._run_prefect)

    def _call_stage(
        self,
        stage: str,
        call: Callable[[], bool],
        *,
        continue_on_error: bool,
    ) -> bool:
        try:
            return call()
        except Exception as error:
            logger.error("失败 {}: {}", stage, format_exception(error))
            traceback.print_exc()
            if not continue_on_error:
                raise
            return False

    def _run(
        self,
        settings: SettingsT | None,
        *,
        overrides: Mapping[str, bool | None] | None,
        only: Iterable[str] | None,
        use_prefect: bool,
    ) -> dict[str, bool]:
        configured = settings if settings is not None else self.settings_loader()
        continue_after_failure = self.continue_on_error(configured)
        results: dict[str, bool] = {}
        for item in self.registry.select(
            configured,
            overrides=overrides,
            only=only,
        ):
            stage_call = item.prefect_call if use_prefect else item.spec.call
            results[item.spec.key] = self._call_stage(
                item.spec.key,
                lambda call=stage_call: call(configured),
                continue_on_error=continue_after_failure,
            )
        return results

    def _run_prefect(
        self,
        settings: SettingsT | None = None,
        overrides: Mapping[str, bool | None] | None = None,
        only: Iterable[str] | None = None,
    ) -> dict[str, bool]:
        return self._run(
            settings,
            overrides=overrides,
            only=only,
            use_prefect=True,
        )

    def run_prefect(
        self,
        settings: SettingsT | None = None,
        *,
        overrides: Mapping[str, bool | None] | None = None,
        only: Iterable[str] | None = None,
    ) -> dict[str, bool]:
        return self._prefect_flow(
            settings=settings,
            overrides=overrides,
            only=only,
        )

    def run_direct(
        self,
        settings: SettingsT | None = None,
        *,
        overrides: Mapping[str, bool | None] | None = None,
        only: Iterable[str] | None = None,
    ) -> dict[str, bool]:
        return self._run(
            settings,
            overrides=overrides,
            only=only,
            use_prefect=False,
        )

    def __call__(
        self,
        settings: SettingsT | None = None,
        *,
        overrides: Mapping[str, bool | None] | None = None,
        only: Iterable[str] | None = None,
        use_prefect: bool = False,
    ) -> dict[str, bool]:
        runner = self.run_prefect if use_prefect else self.run_direct
        return runner(settings, overrides=overrides, only=only)


def build_task_pipeline(
    specs: Iterable[TaskSpec[SettingsT]],
    *,
    settings_loader: SettingsLoader[SettingsT],
    continue_on_error: ContinueOnError[SettingsT],
    flow_name: str,
) -> TaskPipeline[SettingsT]:
    return TaskPipeline(
        TaskRegistry(specs),
        settings_loader=settings_loader,
        continue_on_error=continue_on_error,
        flow_name=flow_name,
    )


__all__ = [
    "TaskPipeline",
    "build_task_pipeline",
]
