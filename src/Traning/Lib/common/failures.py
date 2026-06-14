from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Any


ERROR_DETAIL_KEYS = ("error", "error_type", "error_function", "error_module")


def _error_traceback(error: BaseException) -> TracebackType | None:
    traceback = error.__traceback__
    selected = traceback

    while traceback is not None:
        module_name = str(traceback.tb_frame.f_globals.get("__name__", ""))
        if module_name == "Traning" or module_name.startswith("Traning."):
            selected = traceback
        traceback = traceback.tb_next

    return selected


def callable_location(function: Callable[..., object]) -> tuple[str, str]:
    module_name = str(getattr(function, "__module__", ""))
    function_name = str(
        getattr(function, "__qualname__", getattr(function, "__name__", "<unknown>"))
    )
    return module_name, function_name


def exception_location(error: BaseException) -> tuple[str, str]:
    traceback = _error_traceback(error)
    if traceback is None:
        return "", "<unknown>"

    frame = traceback.tb_frame
    module_name = str(frame.f_globals.get("__name__", ""))
    function_name = str(
        getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
    )
    return module_name, function_name


def failure_detail(
    message: str,
    function: Callable[..., object],
    *,
    error_type: str = "ProcessingStateError",
    **context: Any,
) -> dict[str, Any]:
    module_name, function_name = callable_location(function)
    return {
        **context,
        "error": message,
        "error_type": error_type,
        "error_function": function_name,
        "error_module": module_name,
    }


def exception_detail(error: BaseException, **context: Any) -> dict[str, Any]:
    module_name, function_name = exception_location(error)
    return {
        **context,
        "error": str(error),
        "error_type": type(error).__name__,
        "error_function": function_name,
        "error_module": module_name,
    }


def format_failure(detail: dict[str, Any]) -> str:
    module_name = str(detail.get("error_module", ""))
    function_name = str(detail.get("error_function", "<unknown>"))
    location = f"{module_name}.{function_name}" if module_name else function_name
    return f"函数={location} | {detail.get('error_type', 'Error')}: {detail.get('error', '')}"


def format_exception(error: BaseException) -> str:
    return format_failure(exception_detail(error))


__all__ = [
    "ERROR_DETAIL_KEYS",
    "callable_location",
    "exception_detail",
    "exception_location",
    "failure_detail",
    "format_exception",
    "format_failure",
]
