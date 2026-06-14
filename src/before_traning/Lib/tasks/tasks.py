from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Mapping, TypeVar

from prefect import task


SettingsT = TypeVar("SettingsT")
TaskCall = Callable[[SettingsT], bool]


def require_success(stage: str, success: bool) -> bool:
    if not success:
        raise RuntimeError(f"{stage} 存在失败项，详情见处理状态数据库")
    return True


@dataclass(frozen=True)
class TaskSpec(Generic[SettingsT]):
    key: str
    call: TaskCall[SettingsT]
    override_key: str
    enabled_path: tuple[str, ...]
    retries: int = 0
    prefect_name: str | None = None

    def default_enabled(self, settings: SettingsT) -> bool:
        value: object = settings
        for attribute in self.enabled_path:
            value = getattr(value, attribute)
        return bool(value)


@dataclass(frozen=True)
class RegisteredTask(Generic[SettingsT]):
    spec: TaskSpec[SettingsT]
    prefect_call: TaskCall[SettingsT]


def _build_prefect_task(spec: TaskSpec[SettingsT]) -> TaskCall[SettingsT]:
    def run_registered_task(settings: SettingsT) -> bool:
        return require_success(spec.key, spec.call(settings))

    run_registered_task.__name__ = f"{spec.key}_task"
    run_registered_task.__qualname__ = run_registered_task.__name__
    return task(
        name=spec.prefect_name or spec.key,
        retries=spec.retries,
    )(run_registered_task)


class TaskRegistry(Generic[SettingsT]):
    def __init__(self, specs: Iterable[TaskSpec[SettingsT]]):
        registered: list[RegisteredTask[SettingsT]] = []
        keys: set[str] = set()
        override_keys: set[str] = set()
        for spec in specs:
            if spec.key in keys:
                raise ValueError(f"重复 task key: {spec.key}")
            if spec.override_key in override_keys:
                raise ValueError(f"重复 task override_key: {spec.override_key}")
            keys.add(spec.key)
            override_keys.add(spec.override_key)
            registered.append(
                RegisteredTask(
                    spec=spec,
                    prefect_call=_build_prefect_task(spec),
                )
            )

        if not registered:
            raise ValueError("task 注册表不能为空")
        self._registered = tuple(registered)
        self._by_key = {item.spec.key: item for item in self._registered}
        self._override_keys = frozenset(override_keys)

    @property
    def registered(self) -> tuple[RegisteredTask[SettingsT], ...]:
        return self._registered

    def select(
        self,
        settings: SettingsT,
        *,
        overrides: Mapping[str, bool | None] | None = None,
        only: Iterable[str] | None = None,
    ) -> tuple[RegisteredTask[SettingsT], ...]:
        override_values = dict(overrides or {})
        unknown_overrides = set(override_values) - self._override_keys
        if unknown_overrides:
            raise KeyError(
                f"未知 task override: {', '.join(sorted(unknown_overrides))}"
            )

        selected_keys = set(only) if only is not None else None
        if selected_keys is not None:
            unknown_keys = selected_keys - self._by_key.keys()
            if unknown_keys:
                raise KeyError(
                    f"未知 task key: {', '.join(sorted(unknown_keys))}"
                )

        selected: list[RegisteredTask[SettingsT]] = []
        for item in self._registered:
            spec = item.spec
            if selected_keys is not None and spec.key not in selected_keys:
                continue
            override = override_values.get(spec.override_key)
            enabled = spec.default_enabled(settings) if override is None else override
            if enabled:
                selected.append(item)
        return tuple(selected)


__all__ = [
    "RegisteredTask",
    "TaskCall",
    "TaskRegistry",
    "TaskSpec",
    "require_success",
]
