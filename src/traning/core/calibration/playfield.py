from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from traning.conf import DataSplit, Settings
from traning.core.dataset_import import build_dataset
from traning.lib.coordinates import transform_from_settings_or_sample
from traning.lib.data.models import SegmentRecord


@dataclass(frozen=True)
class CalibrationPoint:
    sample_key: str
    source_index: int | None
    timestamp_ms: float
    osu_xy: tuple[float, float]
    detected_xy: tuple[float, float]
    search_xy: tuple[float, float]
    detected_radius_px: float
    search_error_px: float
    reprojection_error_px: float | None = None
    inlier: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_key": self.sample_key,
            "source_index": self.source_index,
            "timestamp_ms": self.timestamp_ms,
            "osu_xy": list(self.osu_xy),
            "detected_xy": list(self.detected_xy),
            "search_xy": list(self.search_xy),
            "detected_radius_px": self.detected_radius_px,
            "search_error_px": self.search_error_px,
            "reprojection_error_px": self.reprojection_error_px,
            "inlier": self.inlier,
        }


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]]
    inverse_matrix: tuple[tuple[float, float, float], tuple[float, float, float]]
    sample_count: int
    inlier_count: int
    outlier_count: int
    mean_error_px: float
    median_error_px: float
    max_error_px: float
    points: tuple[CalibrationPoint, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": "osu-playfield-calibration-v1",
            "status": self.status,
            "matrix_type": "affine_2x3_osu_to_training_frame",
            "matrix": [list(row) for row in self.matrix],
            "inverse_matrix": [list(row) for row in self.inverse_matrix],
            "sample_count": self.sample_count,
            "inlier_count": self.inlier_count,
            "outlier_count": self.outlier_count,
            "mean_error_px": self.mean_error_px,
            "median_error_px": self.median_error_px,
            "max_error_px": self.max_error_px,
            "points": [point.as_dict() for point in self.points],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def calibrate_playfield_transform(
    settings: Settings,
    *,
    split: DataSplit = "train",
    max_records: int | None = None,
    search_radius_px: int = 150,
    max_search_error_px: float = 90.0,
    ransac_threshold_px: float = 20.0,
    min_inliers: int = 32,
    output_path: Path | None = None,
) -> CalibrationResult:
    dataset = build_dataset(settings, split=split)
    records = dataset.records if max_records is None else dataset.records[:max_records]
    points: list[CalibrationPoint] = []
    for record in records:
        points.extend(
            _detect_record_points(
                settings,
                record,
                search_radius_px=search_radius_px,
                max_search_error_px=max_search_error_px,
            )
        )
    if len(points) < 3:
        raise ValueError("calibration requires at least three detected points")
    src = np.array([point.osu_xy for point in points], dtype=np.float32)
    dst = np.array([point.detected_xy for point in points], dtype=np.float32)
    matrix, inlier_mask = cv2.estimateAffine2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_px,
        maxIters=5000,
        confidence=0.99,
        refineIters=20,
    )
    if matrix is None or inlier_mask is None:
        raise ValueError("failed to fit affine calibration matrix")
    predicted = src @ matrix[:, :2].T + matrix[:, 2]
    errors = np.linalg.norm(predicted - dst, axis=1)
    mask = inlier_mask.reshape(-1).astype(bool)
    inlier_errors = errors[mask]
    inlier_count = int(mask.sum())
    status = "calibrated" if inlier_count >= min_inliers else "insufficient_inliers"
    inverse = _invert_affine(matrix)
    result_points = tuple(
        CalibrationPoint(
            sample_key=point.sample_key,
            source_index=point.source_index,
            timestamp_ms=point.timestamp_ms,
            osu_xy=point.osu_xy,
            detected_xy=point.detected_xy,
            search_xy=point.search_xy,
            detected_radius_px=point.detected_radius_px,
            search_error_px=point.search_error_px,
            reprojection_error_px=float(errors[index]),
            inlier=bool(mask[index]),
        )
        for index, point in enumerate(points)
    )
    result = CalibrationResult(
        status=status,
        matrix=_matrix_tuple(matrix),
        inverse_matrix=_matrix_tuple(inverse),
        sample_count=len(points),
        inlier_count=inlier_count,
        outlier_count=len(points) - inlier_count,
        mean_error_px=float(inlier_errors.mean()) if inlier_count else float("inf"),
        median_error_px=float(np.median(inlier_errors)) if inlier_count else float("inf"),
        max_error_px=float(inlier_errors.max()) if inlier_count else float("inf"),
        points=result_points,
    )
    if output_path is not None:
        result.write_json(output_path)
    return result


def _detect_record_points(
    settings: Settings,
    record: SegmentRecord,
    *,
    search_radius_px: int,
    max_search_error_px: float,
) -> tuple[CalibrationPoint, ...]:
    capture = cv2.VideoCapture(str(record.video_path))
    if not capture.isOpened():
        return ()
    points: list[CalibrationPoint] = []
    try:
        for item in record.annotation.hit_objects:
            if item.x is None or item.y is None:
                continue
            if "circle" not in item.type.lower():
                continue
            timestamp_ms = float(item.start_ms)
            if timestamp_ms < 0 or timestamp_ms > record.annotation.duration_ms:
                continue
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            ok, frame_bgr = capture.read()
            if not ok:
                continue
            frame_height, frame_width = frame_bgr.shape[:2]
            search_transform, _ = transform_from_settings_or_sample(
                settings,
                {"preprocessing_metadata": record.preprocessing_metadata},
                frame_width=frame_width,
                frame_height=frame_height,
            )
            search_xy = search_transform.osu_to_video(float(item.x), float(item.y))
            detected = _detect_circle_near(
                frame_bgr,
                search_xy=search_xy,
                search_radius_px=search_radius_px,
                expected_radius_px=search_transform.osu_radius_to_video(
                    record.annotation.difficulty.circle_radius_osu_pixels
                ),
            )
            if detected is None:
                continue
            detected_x, detected_y, radius_px, search_error = detected
            if search_error > max_search_error_px:
                continue
            points.append(
                CalibrationPoint(
                    sample_key=record.key,
                    source_index=item.source_index,
                    timestamp_ms=timestamp_ms,
                    osu_xy=(float(item.x), float(item.y)),
                    detected_xy=(float(detected_x), float(detected_y)),
                    search_xy=(float(search_xy[0]), float(search_xy[1])),
                    detected_radius_px=float(radius_px),
                    search_error_px=float(search_error),
                )
            )
    finally:
        capture.release()
    return tuple(points)


def _detect_circle_near(
    frame_bgr: np.ndarray,
    *,
    search_xy: tuple[float, float],
    search_radius_px: int,
    expected_radius_px: float,
) -> tuple[float, float, float, float] | None:
    height, width = frame_bgr.shape[:2]
    x, y = search_xy
    x0 = max(0, int(round(x - search_radius_px)))
    y0 = max(0, int(round(y - search_radius_px)))
    x1 = min(width, int(round(x + search_radius_px)))
    y1 = min(height, int(round(y + search_radius_px)))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=80,
        param2=20,
        minRadius=20,
        maxRadius=130,
    )
    if circles is None:
        return None
    best: tuple[float, float, float, float, float] | None = None
    for local_x, local_y, radius in np.round(circles[0]).astype(int):
        detected_x = float(local_x + x0)
        detected_y = float(local_y + y0)
        radius_px = float(radius)
        search_error = math.hypot(detected_x - x, detected_y - y)
        score = search_error + abs(radius_px - expected_radius_px) * 0.25
        if best is None or score < best[0]:
            best = (score, detected_x, detected_y, radius_px, search_error)
    if best is None:
        return None
    _, detected_x, detected_y, radius_px, search_error = best
    return detected_x, detected_y, radius_px, search_error


def _invert_affine(matrix: np.ndarray) -> np.ndarray:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    determinant = a * e - b * d
    if abs(float(determinant)) <= 1e-9:
        raise ValueError("affine matrix is not invertible")
    return np.array(
        [
            [e / determinant, -b / determinant, (b * f - e * c) / determinant],
            [-d / determinant, a / determinant, (d * c - a * f) / determinant],
        ],
        dtype=np.float64,
    )


def _matrix_tuple(
    matrix: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
    )


__all__ = ["CalibrationResult", "calibrate_playfield_transform"]
