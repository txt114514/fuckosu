from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from package.coordinates import (
    COORDINATE_TRANSFORM_VERSION,
    CoordinateTransformSpec,
    OsuVideoTransform,
    PlayfieldRect,
)


def transform_from_settings_or_sample(
    settings: Any | None,
    sample: Mapping[str, Any] | None = None,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[OsuVideoTransform, CoordinateTransformSpec]:
    """Resolve the explicit playfield transform for original video pixels."""

    sample_spec = _sample_transform_spec(sample)
    if sample_spec is not None:
        return OsuVideoTransform.from_rect(sample_spec.rect), sample_spec

    config = getattr(settings, "coordinate_transform", None)
    if config is not None and getattr(config, "mode", None) == "explicit_rect":
        rect = getattr(config, "playfield_rect", None)
        if rect is None:
            raise ValueError("explicit coordinate transform is missing playfield_rect")
        playfield = PlayfieldRect(
            left=float(rect.left),
            top=float(rect.top),
            width=float(rect.width),
            height=float(rect.height),
        )
        spec = CoordinateTransformSpec(
            version=COORDINATE_TRANSFORM_VERSION,
            rect=playfield,
            source="settings.explicit_rect",
        )
        return OsuVideoTransform.from_rect(playfield), spec

    if config is None:
        if frame_width is None or frame_height is None:
            raise ValueError("coordinate transform requires settings or frame size")
        transform = OsuVideoTransform.fit_centered(frame_width, frame_height)
        return transform, transform.spec(source="legacy_centered_unconfigured")
    if getattr(config, "mode", None) != "legacy_centered":
        raise ValueError(
            "playfield calibration is required; set coordinate_transform.mode="
            "explicit_rect or explicitly opt into legacy_centered"
        )
    if frame_width is None or frame_height is None:
        raise ValueError("legacy_centered transform requires frame_width/frame_height")
    transform = OsuVideoTransform.fit_centered(frame_width, frame_height)
    return transform, transform.spec(source="settings.legacy_centered")


def _sample_transform_spec(
    sample: Mapping[str, Any] | None,
) -> CoordinateTransformSpec | None:
    if sample is None:
        return None
    raw = sample.get("coordinate_transform") or sample.get("playfield_transform")
    if not isinstance(raw, Mapping):
        rect = sample.get("playfield_rect")
        if not isinstance(rect, Mapping):
            return None
        raw = {"version": COORDINATE_TRANSFORM_VERSION, "rect": rect}
    version = str(raw.get("version") or "")
    if version != COORDINATE_TRANSFORM_VERSION:
        raise ValueError(
            f"unsupported coordinate transform version: {version or '<missing>'}"
        )
    rect = raw.get("rect") or raw.get("playfield_rect")
    if not isinstance(rect, Mapping):
        raise ValueError("coordinate transform metadata is missing rect")
    return CoordinateTransformSpec(
        version=version,
        rect=PlayfieldRect.from_mapping(rect),
        source=str(raw.get("source") or "sample"),
    )


__all__ = ["transform_from_settings_or_sample"]
