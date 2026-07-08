from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from package.coordinates import (
    AffineOsuVideoTransform,
    COORDINATE_TRANSFORM_VERSION,
    CoordinateTransformChain,
    CoordinateTransformSpec,
    ImageSize,
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
    """Resolve the playfield transform in training-frame pixels."""

    sample_spec = _sample_transform_spec(sample)
    if sample_spec is not None:
        if sample_spec.matrix is not None:
            return AffineOsuVideoTransform(sample_spec.matrix), sample_spec
        return OsuVideoTransform.from_rect(sample_spec.rect), sample_spec

    config = getattr(settings, "coordinate_transform", None)
    if config is not None and getattr(config, "mode", None) == "affine_matrix":
        transform = AffineOsuVideoTransform.from_rows(getattr(config, "matrix", None))
        return transform, transform.spec(source="settings.affine_matrix", status="calibrated")

    chain = coordinate_chain_from_settings_or_sample(
        settings,
        sample,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if chain is not None:
        return chain.to_frame_transform(), chain.spec()

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


def coordinate_chain_from_settings_or_sample(
    settings: Any | None,
    sample: Mapping[str, Any] | None = None,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> CoordinateTransformChain | None:
    """Build the shared osu -> source -> crop -> training-frame chain."""

    config = getattr(settings, "coordinate_transform", None)
    if config is None:
        return None
    mode = getattr(config, "mode", None)
    if mode == "explicit_rect":
        rect = getattr(config, "playfield_rect", None)
        if rect is None:
            raise ValueError("explicit coordinate transform is missing playfield_rect")
        size = ImageSize(
            width=float(frame_width or getattr(settings.input, "width", 1)),
            height=float(frame_height or getattr(settings.input, "height", 1)),
        )
        return CoordinateTransformChain(
            source_size=size,
            crop_rect=PlayfieldRect(0.0, 0.0, size.width, size.height),
            resized_size=size,
            playfield_source_rect=_rect_from_object(rect),
            source="settings.explicit_rect",
            status="unresolved",
        )
    if mode == "explicit_source_rect":
        rect = getattr(config, "playfield_rect", None)
        if rect is None:
            raise ValueError("explicit source transform is missing playfield_rect")
        metadata = _preprocessing_metadata(sample)
        crop = _metadata_crop_rect(metadata) or _config_crop_rect(config)
        if crop is None:
            raise ValueError("explicit source transform is missing crop_rect")
        source_size = _metadata_source_size(metadata) or ImageSize(
            width=max(crop.left + crop.width, crop.width),
            height=max(crop.top + crop.height, crop.height),
        )
        resized_size = ImageSize(
            width=float(frame_width or getattr(settings.input, "width", crop.width)),
            height=float(frame_height or getattr(settings.input, "height", crop.height)),
        )
        return CoordinateTransformChain(
            source_size=source_size,
            crop_rect=crop,
            resized_size=resized_size,
            playfield_source_rect=_rect_from_object(rect),
            source=(
                "settings.explicit_source_rect+sample.preprocessing_metadata"
                if metadata is not None
                else "settings.explicit_source_rect"
            ),
            status="unresolved",
        )
    return None


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
        transform_status=str(raw.get("transform_status") or "configured"),
        source_size=_size_from_mapping(raw.get("source_size")),
        crop_rect=_rect_from_mapping(raw.get("crop_rect")),
        resized_size=_size_from_mapping(raw.get("resized_size")),
        playfield_source_rect=_rect_from_mapping(raw.get("playfield_source_rect")),
        chain=raw.get("chain") if isinstance(raw.get("chain"), Mapping) else None,
        matrix=_matrix_from_value(raw.get("matrix")),
    )


def _rect_from_object(value: Any) -> PlayfieldRect:
    return PlayfieldRect(
        left=float(value.left),
        top=float(value.top),
        width=float(value.width),
        height=float(value.height),
    )


def _rect_from_mapping(value: object) -> PlayfieldRect | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return PlayfieldRect.from_mapping(value)
    except (TypeError, ValueError):
        return None


def _size_from_mapping(value: object) -> ImageSize | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return ImageSize.from_mapping(value)
    except (TypeError, ValueError):
        return None


def _matrix_from_value(
    value: object,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return (
            (float(value[0][0]), float(value[0][1]), float(value[0][2])),
            (float(value[1][0]), float(value[1][1]), float(value[1][2])),
        )
    except (TypeError, ValueError, IndexError):
        return None


def _config_crop_rect(config: Any) -> PlayfieldRect | None:
    crop = getattr(config, "crop_rect", None)
    return None if crop is None else _rect_from_object(crop)


def _preprocessing_metadata(sample: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if sample is None:
        return None
    metadata = sample.get("preprocessing_metadata")
    return metadata if isinstance(metadata, Mapping) else None


def _metadata_crop_rect(metadata: Mapping[str, Any] | None) -> PlayfieldRect | None:
    if metadata is None:
        return None
    return _rect_from_mapping(metadata.get("crop_rect"))


def _metadata_source_size(metadata: Mapping[str, Any] | None) -> ImageSize | None:
    if metadata is None:
        return None
    return _size_from_mapping(metadata.get("source_size"))


__all__ = [
    "coordinate_chain_from_settings_or_sample",
    "transform_from_settings_or_sample",
]
