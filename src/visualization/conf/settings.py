from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from visualization.conf.defaults import (
    DEFAULT_AUTO_PAGE_SECONDS,
    DEFAULT_COMPACT_TERMINAL_HEIGHT,
    DEFAULT_CRITICAL_VRAM_RATIO,
    DEFAULT_LANGUAGE,
    DEFAULT_PLAIN_INTERVAL_SECONDS,
    DEFAULT_RECENT_EVENT_LIMIT,
    DEFAULT_REFRESH_PER_SECOND,
    DEFAULT_WARNING_VRAM_RATIO,
    NARROW_TERMINAL_WIDTH,
)

ProgressUIMode = Literal["auto", "rich", "plain", "off"]


@dataclass(frozen=True)
class DashboardSettings:
    mode: ProgressUIMode = "auto"
    language: str = DEFAULT_LANGUAGE
    refresh_per_second: float = DEFAULT_REFRESH_PER_SECOND
    plain_interval_seconds: float = DEFAULT_PLAIN_INTERVAL_SECONDS
    recent_event_limit: int = DEFAULT_RECENT_EVENT_LIMIT
    warning_vram_ratio: float = DEFAULT_WARNING_VRAM_RATIO
    critical_vram_ratio: float = DEFAULT_CRITICAL_VRAM_RATIO
    compact_terminal_height: int = DEFAULT_COMPACT_TERMINAL_HEIGHT
    auto_page_seconds: float = DEFAULT_AUTO_PAGE_SECONDS
    narrow_terminal_width: int = NARROW_TERMINAL_WIDTH

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "rich", "plain", "off"}:
            raise ValueError("界面模式必须是 auto、rich、plain 或 off")
        if self.refresh_per_second <= 0:
            raise ValueError("刷新频率必须为正数")
        if self.plain_interval_seconds <= 0:
            raise ValueError("纯文本刷新间隔必须为正数")
        if self.recent_event_limit <= 0:
            raise ValueError("最近事件数量上限必须为正数")
        if self.compact_terminal_height <= 0:
            raise ValueError("紧凑终端高度阈值必须为正数")
        if self.auto_page_seconds <= 0:
            raise ValueError("分页兼容间隔必须为正数")
