from __future__ import annotations

import unittest

from package import (
    ArtifactFileRef,
    CandidateCacheFrameRef,
    CheckpointRef,
    CoordinateSpace,
    DataSplit,
    DecisionFrameRecord,
    ErrorDomain,
    EvaluationOutcome,
    FrameRef,
    OsuHitObject,
    Point2D,
    PredictionAction,
    PredictionEvent,
    Rect2D,
    SegmentCategory,
    SegmentManifestEntry,
    SegmentRef,
    SpatialCandidateRef,
    TemporalTargetRef,
    TrialRef,
    SearchMethod,
    VersionedArtifactRef,
)


class ContractTests(unittest.TestCase):
    def test_geometry_contracts_validate_space_and_bounds(self) -> None:
        point = Point2D(10.0, 20.0, CoordinateSpace.VIDEO)
        rect = Rect2D(0.0, 0.0, 100.0, 100.0, "video")

        self.assertTrue(rect.contains(point))
        self.assertEqual(point.as_dict()["space"], "video")

    def test_evaluation_contract_round_trips_from_mapping(self) -> None:
        outcome = EvaluationOutcome.from_mapping(
            {
                "frame": {
                    "sample_key": "item_000001/segment_000001",
                    "frame_index": 12,
                    "timestamp_ms": 200.0,
                },
                "passed": False,
                "primary_error": "temporal",
                "error_tags": ["late_click"],
                "metrics": {"quality_score": 0.5},
                "prediction": {
                    "action": "press",
                    "point": {"x": 256.0, "y": 192.0, "space": "osu"},
                    "time_ms": 225.0,
                    "candidate_id": 3,
                    "confidence": 0.8,
                },
            }
        )

        self.assertEqual(outcome.primary_error, ErrorDomain.TEMPORAL)
        self.assertEqual(outcome.prediction.action, PredictionAction.PRESS)
        self.assertEqual(outcome.as_dict()["prediction"]["point"]["space"], "osu")

    def test_artifact_contracts_normalize_nested_files(self) -> None:
        artifact = VersionedArtifactRef(
            artifact_id="artifact-1",
            schema_version="v1",
            files=(
                ArtifactFileRef("manifest", "manifest.json", size_bytes=10),
                {"role": "weights", "path": "model.pt", "sha256": "abc"},
            ),
        )

        self.assertEqual(len(artifact.files), 2)
        self.assertEqual(artifact.files[1].role, "weights")
        self.assertEqual(artifact.as_dict()["files"][0]["role"], "manifest")

    def test_invalid_contract_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            FrameRef("", -1)
        with self.assertRaises(ValueError):
            PredictionEvent(action="press", confidence=2.0)

    def test_osu_contracts_cover_circle_slider_and_spinner(self) -> None:
        circle = OsuHitObject.circle(
            "circle-1",
            start_ms=1000.0,
            x=256.0,
            y=192.0,
        )
        slider = OsuHitObject.slider(
            "slider-1",
            start_ms=1000.0,
            end_ms=1500.0,
            path=((0.0, 0.0), (128.0, 128.0)),
            pixel_length=180.0,
        )
        spinner = OsuHitObject.spinner(
            "spinner-1",
            start_ms=2000.0,
            end_ms=3000.0,
        )

        self.assertEqual(circle.object_type, "circle")
        self.assertEqual(slider.path[1].as_tuple(), (128.0, 128.0))
        self.assertEqual(spinner.as_dict()["object_type"], "spinner")

    def test_dataset_contracts_describe_segment_manifest_rows(self) -> None:
        segment = SegmentRef(
            sample_key="item_000001/segment_000001",
            item_name="item_000001",
            category=SegmentCategory.SINGLE_POINT,
            dimension="atomic",
            video_path="video.mp4",
            annotation_path="beatmap.json",
        )
        entry = SegmentManifestEntry(segment=segment, split=DataSplit.TRAIN)

        self.assertEqual(entry.as_dict()["split"], "train")
        self.assertEqual(entry.segment.category, SegmentCategory.SINGLE_POINT)

    def test_candidate_contracts_cover_cache_and_decision_records(self) -> None:
        candidate = SpatialCandidateRef(
            candidate_id=1,
            point={"x": 100.0, "y": 120.0, "space": "video"},
            score=0.9,
            object_type="circle",
            embedding=(0.1, 0.2),
        )
        frame = CandidateCacheFrameRef(
            version="cache-v1",
            sample_key="sample-a",
            frame_index=3,
            timestamp_ms=50.0,
            candidates=(candidate,),
            temporal_target=TemporalTargetRef(action="press", candidate_id=1),
        )
        decision = DecisionFrameRecord(
            version="decision-v1",
            sample_key="sample-a",
            frame_index=3,
            action="press",
            action_probability=0.75,
            candidate_id=1,
        )

        self.assertEqual(frame.candidates[0].point.space, CoordinateSpace.VIDEO)
        self.assertEqual(decision.as_dict()["action"], "press")

    def test_experiment_contracts_describe_trials_and_checkpoints(self) -> None:
        trial = TrialRef(
            trial_id="trial-1",
            experiment_name="exp",
            seed=2026,
            search_method=SearchMethod.TPE,
            budget_steps=100,
            metrics={"quality_score": 0.8},
        )
        checkpoint = CheckpointRef(
            checkpoint_id="ckpt-1",
            trial_id=trial.trial_id,
            path="temporal_model.pt",
            global_step=100,
        )

        self.assertEqual(trial.as_dict()["search_method"], "tpe")
        self.assertEqual(checkpoint.trial_id, "trial-1")


if __name__ == "__main__":
    unittest.main()
