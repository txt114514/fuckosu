from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from visualization.conf.messages import display_text


def apply_display_overrides(renderable: Any) -> Any:
    """Apply display-name overrides at the final Rich renderable boundary."""

    return _apply(renderable, seen=set())


def _apply(value: Any, *, seen: set[int]) -> Any:
    if isinstance(value, str):
        return display_text(value)
    if isinstance(value, Text):
        return _text_with_overrides(value)
    identity = id(value)
    if identity in seen:
        return value
    seen.add(identity)
    if isinstance(value, Panel):
        value.renderable = _apply(value.renderable, seen=seen)
        value.title = _apply_optional(value.title, seen=seen)
        value.subtitle = _apply_optional(value.subtitle, seen=seen)
        return value
    if isinstance(value, Table):
        return _table_with_overrides(value, seen=seen)
    if isinstance(value, Group):
        return Group(
            *(_apply(item, seen=seen) for item in getattr(value, "_renderables", ())),
            fit=value.fit,
        )
    return value


def _apply_optional(value: Any, *, seen: set[int]) -> Any:
    if value is None:
        return None
    return _apply(value, seen=seen)


def _text_with_overrides(value: Text) -> Text:
    translated = display_text(value.plain)
    if translated == value.plain:
        return value
    return Text(translated, style=value.style, justify=value.justify)


def _table_with_overrides(table: Table, *, seen: set[int]) -> Table:
    table.title = _apply_optional(table.title, seen=seen)
    table.caption = _apply_optional(table.caption, seen=seen)
    for column in table.columns:
        column.header = _apply(column.header, seen=seen)
        column.footer = _apply_optional(column.footer, seen=seen)
        column._cells = [_apply(cell, seen=seen) for cell in column._cells]
    return table


__all__ = ["apply_display_overrides"]
