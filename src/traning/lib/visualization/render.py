from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from package import OsuVideoTransform, sample_slider_path


VISIBLE_COLOR = (46, 220, 255)
TARGET_COLOR = (255, 72, 72)
PATH_COLOR = (255, 206, 64)
PLAYFIELD_COLOR = (120, 255, 120)
TEXT_COLOR = (255, 255, 255)


def _image_from_tensor(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
        raise ValueError("image tensor must use CHW layout")
    data = image.detach().cpu()
    if data.is_floating_point():
        data = data.clamp(0, 1).mul(255)
    array = data.to(torch.uint8).permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return Image.fromarray(array[:, :, :3], mode="RGB")


def _point(
    transform: OsuVideoTransform,
    x: float,
    y: float,
) -> tuple[int, int]:
    video_x, video_y = transform.osu_to_video(x, y)
    return round(video_x), round(video_y)


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    color: tuple[int, int, int],
    size: int = 12,
    width: int = 3,
) -> None:
    x, y = point
    draw.line((x - size, y, x + size, y), fill=color, width=width)
    draw.line((x, y - size, x, y + size), fill=color, width=width)


def _is_target(
    hit_object: Mapping[str, Any],
    target_source_index: int | None,
) -> bool:
    return (
        target_source_index is not None
        and hit_object.get("source_index") == target_source_index
    )


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    hit_object: Mapping[str, Any],
    transform: OsuVideoTransform,
    radius: int,
    target_source_index: int | None,
) -> tuple[int, int] | None:
    x = hit_object.get("x")
    y = hit_object.get("y")
    if x is None or y is None:
        return None
    center = _point(transform, float(x), float(y))
    color = (
        TARGET_COLOR
        if _is_target(hit_object, target_source_index)
        else VISIBLE_COLOR
    )
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        outline=color,
        width=4,
    )
    _draw_cross(draw, center, color)
    return center


def _draw_slider(
    draw: ImageDraw.ImageDraw,
    hit_object: Mapping[str, Any],
    transform: OsuVideoTransform,
    radius: int,
    target_source_index: int | None,
) -> tuple[int, int] | None:
    raw_path = tuple(
        (float(point[0]), float(point[1]))
        for point in hit_object.get("path", ())
    )
    if not raw_path:
        return None
    try:
        path = sample_slider_path(
            raw_path,
            curve_type=str(hit_object.get("curve_type", "L")),
            pixel_length=hit_object.get("pixel_length"),
        )
    except (TypeError, ValueError):
        path = raw_path
    video_path = [_point(transform, x, y) for x, y in path]
    color = (
        TARGET_COLOR
        if _is_target(hit_object, target_source_index)
        else PATH_COLOR
    )
    if len(video_path) >= 2:
        draw.line(
            video_path,
            fill=color,
            width=max(6, radius // 5),
            joint="curve",
        )
        draw.line(
            video_path,
            fill=(255, 255, 255),
            width=3,
            joint="curve",
        )
    head = video_path[0]
    draw.ellipse(
        (
            head[0] - radius,
            head[1] - radius,
            head[0] + radius,
            head[1] + radius,
        ),
        outline=color,
        width=4,
    )
    _draw_cross(draw, head, color)
    return head


def render_annotated_frame(
    sample: Mapping[str, Any],
    *,
    target_source_index: int | None = None,
    include_all_objects: bool = False,
    predicted_osu_xy: tuple[float, float] | None = None,
    metadata_lines: Sequence[str] = (),
) -> Image.Image:
    image = _image_from_tensor(sample["image"])
    width, height = image.size
    transform = OsuVideoTransform.fit_centered(width, height)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    left = round(transform.playfield_left)
    top = round(transform.playfield_top)
    right = round(transform.playfield_left + transform.playfield_width)
    bottom = round(transform.playfield_top + transform.playfield_height)
    draw.rectangle((left, top, right, bottom), outline=PLAYFIELD_COLOR, width=2)

    radius = max(
        3,
        round(
            transform.osu_radius_to_video(
                float(sample["circle_radius_osu_pixels"])
            )
        ),
    )
    objects: Sequence[Mapping[str, Any]] = sample[
        "hit_objects" if include_all_objects else "visible_hit_objects"
    ]
    target_point: tuple[int, int] | None = None
    spinner_target = False
    for hit_object in objects:
        object_type = hit_object.get("type")
        if object_type == "circle":
            point = _draw_circle(
                draw,
                hit_object,
                transform,
                radius,
                target_source_index,
            )
        elif object_type == "slider":
            point = _draw_slider(
                draw,
                hit_object,
                transform,
                radius,
                target_source_index,
            )
        elif object_type == "spinner":
            point = None
            spinner_target = _is_target(hit_object, target_source_index)
            draw.ellipse(
                (left, top, right, bottom),
                outline=TARGET_COLOR if spinner_target else VISIBLE_COLOR,
                width=5,
            )
        else:
            point = None
        if _is_target(hit_object, target_source_index):
            target_point = point

    lines = [
        f"sample={sample['sample_key']}",
        f"frame={sample['frame_index']} t={sample['timestamp_ms']:.3f}ms",
        f"visible={len(sample['visible_hit_objects'])}",
    ]
    if target_point is not None:
        osu_point = transform.video_to_osu(*target_point)
        lines.append(
            "target "
            f"video=({target_point[0]}, {target_point[1]}) "
            f"osu=({osu_point[0]:.2f}, {osu_point[1]:.2f})"
        )
    elif spinner_target:
        lines.append("target=spinner")
    if predicted_osu_xy is not None:
        predicted_point = _point(
            transform,
            predicted_osu_xy[0],
            predicted_osu_xy[1],
        )
        _draw_cross(draw, predicted_point, (255, 80, 255), size=16, width=4)
        lines.append(
            "prediction "
            f"video=({predicted_point[0]}, {predicted_point[1]}) "
            f"osu=({predicted_osu_xy[0]:.2f}, {predicted_osu_xy[1]:.2f})"
        )
    lines.extend(str(line) for line in metadata_lines)

    text = "\n".join(lines)
    text_box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    box_width = text_box[2] - text_box[0] + 16
    box_height = text_box[3] - text_box[1] + 16
    draw.rectangle((8, 8, 8 + box_width, 8 + box_height), fill=(0, 0, 0))
    draw.multiline_text((16, 16), text, fill=TEXT_COLOR, font=font, spacing=4)
    return image


def save_annotated_frame(image: Image.Image, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


__all__ = ["render_annotated_frame", "save_annotated_frame"]
