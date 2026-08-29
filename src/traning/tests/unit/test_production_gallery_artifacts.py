"""验证 production PNG 按 canonical 单帧事件分类并以 manifest 收口。"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from PIL import Image
from package import AffineOsuVideoTransform

from traning.contracts import RuntimeFrame
from traning.data import FrameCoordinateTransform
from traning.evaluation import (
    FramePredictedClick,
    PrimaryError,
    TargetObject,
    build_sequence_evaluation_events,
    score_frame_click_sequence,
)
from traning.infrastructure import read_json_object
from traning.training import (
    PRODUCTION_GALLERY_MANIFEST_FILENAME,
    publish_production_gallery_manifest,
    render_production_sequence_gallery,
)
from traning.visualization import build_gallery_frame_overlay


_WIDTH = 64
_HEIGHT = 48
_SEQUENCE_ID = "item_000001/long_sequence_000008"


def _transform() -> FrameCoordinateTransform:
    """构造像素与 osu! 坐标一致的可逆测试变换。"""

    return FrameCoordinateTransform(
        source_frame_width=_WIDTH,
        source_frame_height=_HEIGHT,
        transform_identity="production-gallery-test",
        transform=AffineOsuVideoTransform(((0.1, 0.0, 0.0), (0.0, 0.1, 0.0))),
    )


def _frame(frame_index: int, timestamp_ms: float) -> RuntimeFrame:
    """生成可由真实 PNG renderer 消费的 packed RGB 原帧。"""

    return RuntimeFrame(
        frame_id=f"frame-{frame_index}",
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        width=_WIDTH,
        height=_HEIGHT,
        image_bytes=b"\x00" * (_WIDTH * _HEIGHT * 3),
    )


def test_production_gallery_classifies_each_real_frame_and_commits_manifest(
    tmp_path: Path,
) -> None:
    """frame 105 失败不得把 frame 36 的命中图拖进 failed。"""

    transform = _transform()
    targets = (
        TargetObject(
            "target-36",
            "circle",
            36.0,
            36.0,
            x=10.0,
            y=10.0,
            source_index=0,
            frame_index=36,
        ),
        TargetObject(
            "target-105",
            "circle",
            105.0,
            105.0,
            x=30.0,
            y=30.0,
            source_index=1,
            frame_index=105,
        ),
    )
    clicks = (
        FramePredictedClick(
            36.0,
            transform.bind_frame_prediction(
                x=1.0,
                y=1.0,
                source_frame_width=_WIDTH,
                source_frame_height=_HEIGHT,
            ),
            frame_index=36,
        ),
    )
    score = score_frame_click_sequence(
        targets,
        clicks,
        coordinate_transform=transform,
        circle_radius=4.0,
    )
    events = build_sequence_evaluation_events(
        _SEQUENCE_ID,
        105,
        score,
    )
    assert tuple((event.frame_index, event.passed) for event in events) == (
        (36, True),
        (105, False),
    )

    directory = tmp_path / "gallery"
    records = render_production_sequence_gallery(
        directory,
        sequence_id=_SEQUENCE_ID,
        frames=(_frame(36, 36.0), _frame(105, 105.0)),
        targets=targets,
        score=score,
        events=events,
        coordinate_transform=transform,
    )

    assert tuple(record.frame_index for record in records) == (36, 105)
    passed_record, failed_record = records
    assert passed_record.passed
    assert passed_record.primary_errors == ()
    assert passed_record.relative_png_path.startswith("passed/none/")
    assert failed_record.passed is False
    assert failed_record.primary_errors == (PrimaryError.DECISION,)
    assert failed_record.relative_png_path.startswith("failed/decision/")
    assert "long_sequence_000008" in failed_record.relative_png_path
    assert ".." not in PurePosixPath(failed_record.relative_png_path).parts
    assert len(PurePosixPath(failed_record.relative_png_path).parts) == 4

    for record in records:
        png_path = directory.joinpath(*PurePosixPath(record.relative_png_path).parts)
        assert png_path.is_file()
        with Image.open(png_path) as image:
            assert image.size == (_WIDTH, _HEIGHT)
            assert image.format == "PNG"

    unresolved_overlay = build_gallery_frame_overlay(
        targets,
        score,
        events,
        transform,
        frame_index=105,
    )
    assert unresolved_overlay.predictions == ()
    assert unresolved_overlay.events[0].primary_error is PrimaryError.DECISION

    manifest_path = publish_production_gallery_manifest(
        directory,
        run_id="production-gallery-run",
        dataset_id=f"dataset-{'a' * 64}",
        trial_index=2,
        transform_fingerprint=transform.transform_fingerprint,
        records=records,
    )
    assert manifest_path == directory / PRODUCTION_GALLERY_MANIFEST_FILENAME
    manifest = read_json_object(manifest_path)
    assert manifest["record_count"] == 2
    manifest_records = manifest["records"]
    assert isinstance(manifest_records, list)
    assert [item["passed"] for item in manifest_records] == [True, False]
    assert manifest_records[1]["primary_errors"] == ["decision"]


def test_gallery_manifest_rejects_tampered_png(tmp_path: Path) -> None:
    """manifest-last 必须在摘要不匹配时拒绝发布完成标记。"""

    transform = _transform()
    target = TargetObject(
        "target-36",
        "circle",
        36.0,
        36.0,
        x=10.0,
        y=10.0,
        frame_index=36,
    )
    click = FramePredictedClick(
        36.0,
        transform.bind_frame_prediction(
            x=1.0,
            y=1.0,
            source_frame_width=_WIDTH,
            source_frame_height=_HEIGHT,
        ),
        frame_index=36,
    )
    score = score_frame_click_sequence(
        (target,),
        (click,),
        coordinate_transform=transform,
        circle_radius=4.0,
    )
    events = build_sequence_evaluation_events(_SEQUENCE_ID, 36, score)
    directory = tmp_path / "gallery"
    records = render_production_sequence_gallery(
        directory,
        sequence_id=_SEQUENCE_ID,
        frames=(_frame(36, 36.0),),
        targets=(target,),
        score=score,
        events=events,
        coordinate_transform=transform,
    )
    png_path = directory.joinpath(*PurePosixPath(records[0].relative_png_path).parts)
    png_path.write_bytes(b"tampered")

    try:
        publish_production_gallery_manifest(
            directory,
            run_id="production-gallery-run",
            dataset_id=f"dataset-{'b' * 64}",
            trial_index=0,
            transform_fingerprint=transform.transform_fingerprint,
            records=records,
        )
    except ValueError as error:
        assert "摘要不匹配" in str(error)
    else:  # pragma: no cover - 必须拒绝损坏文件
        raise AssertionError("损坏 PNG 不得发布 manifest")
    assert not (directory / PRODUCTION_GALLERY_MANIFEST_FILENAME).exists()
