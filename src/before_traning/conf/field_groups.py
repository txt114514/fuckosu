from __future__ import annotations

from types import SimpleNamespace
from typing import Any


TARGET_MANIFEST_FIELDS = ("target_root", "manifest_filename")
VIDEO_TARGET_FIELDS = ("video_root", *TARGET_MANIFEST_FIELDS)
AUDIO_VERIFY_FIELDS = ("audio_filename", "verify_filename")
AV_TUNING_FIELDS = (
    "sample_rate",
    "envelope_hz",
    "refine_hz",
    "refine_search_seconds",
    "music_lowpass_hz",
    "verify_correction_window_ms",
)
CLIP_CROP_FIELDS = (
    "crop_reference_width",
    "crop_reference_height",
    "crop_left",
    "crop_top",
    "crop_right",
    "crop_bottom",
)

PROCESSOR_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "av": (
        *TARGET_MANIFEST_FIELDS,
        *AUDIO_VERIFY_FIELDS,
        "output_filename",
        "status_step",
        *AV_TUNING_FIELDS,
    ),
    "audio_match": (
        *VIDEO_TARGET_FIELDS,
        *AUDIO_VERIFY_FIELDS,
        "top_k",
        "match_status_step",
    ),
    "beatmap_import": (
        "export_dir",
        *TARGET_MANIFEST_FIELDS,
        "audio_filename",
    ),
    "clip_crop": CLIP_CROP_FIELDS,
    "video_match": (
        *VIDEO_TARGET_FIELDS,
        *AUDIO_VERIFY_FIELDS,
        "use_audio_match_experiment",
        *AV_TUNING_FIELDS,
    ),
    "video_package": VIDEO_TARGET_FIELDS,
}

FORWARD_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "audio_match_to_av": (
        *TARGET_MANIFEST_FIELDS,
        *AUDIO_VERIFY_FIELDS,
        *AV_TUNING_FIELDS,
        "video_suffixes",
    ),
    "video_match_to_audio_match": (
        *VIDEO_TARGET_FIELDS,
        *AUDIO_VERIFY_FIELDS,
        "video_suffixes",
        *AV_TUNING_FIELDS,
    ),
}


def group_values(config: SimpleNamespace, group: str) -> tuple[Any, ...]:
    return tuple(getattr(config, name) for name in PROCESSOR_FIELD_GROUPS[group])


def assign_group(target: Any, config: SimpleNamespace, group: str) -> None:
    for name in PROCESSOR_FIELD_GROUPS[group]:
        setattr(target, name, getattr(config, name))


def forward_kwargs(source: Any, group: str) -> dict[str, Any]:
    return {name: getattr(source, name) for name in FORWARD_FIELD_GROUPS[group]}
