from __future__ import annotations

import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch

from before_traning.conf import Settings
from package.dataset_split import (
    DatasetSplitManifest,
    DatasetSplitSyncResult,
    SplitRatios,
)
from package.dataset_split.models import DatasetSplitItem
from package.checks import StartupCheckReport, StartupCheckResult
from start.flow import StartupFlowConfig, run_startup_flow
from start.samples import inspect_before_training_samples


def _settings(root: Path) -> Settings:
    return Settings(
        file_management={
            "export_dir": root / "exports",
            "target_root": root / "match-completed_package",
            "video_root": root / "video_package",
            "segment_root": root / "video_segments",
        }
    )


def _write_osz(path: Path, *, osu_name: str = "Song Normal.osu") -> None:
    osu_text = "\n".join(
        (
            "osu file format v14",
            "",
            "[General]",
            "AudioFilename: audio.mp3",
            "",
            "[HitObjects]",
        )
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(osu_name, osu_text)
        archive.writestr("audio.mp3", b"fake mp3 bytes")


def _write_manifest_db(
    target_root: Path,
    *,
    folder_name: str,
    source_name: str,
    source_osz_name: str,
    source_mtime_ns: int,
    active: bool = True,
) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target_root / ".package_manifest.sqlite") as connection:
        connection.execute(
            """
            CREATE TABLE package_manifest_item (
                id INTEGER PRIMARY KEY,
                folder_name TEXT,
                source_name TEXT,
                sequence INTEGER,
                osu_filename TEXT,
                source_osz_name TEXT,
                source_mtime_ns INTEGER,
                difficulty_value REAL,
                active INTEGER
            )
            """
        )
        connection.execute(
            """
            INSERT INTO package_manifest_item (
                folder_name, source_name, sequence, osu_filename,
                source_osz_name, source_mtime_ns, active
            )
            VALUES (?, ?, 1, ?, ?, ?, ?)
            """,
            (
                folder_name,
                source_name,
                f"{source_name}.osu",
                source_osz_name,
                source_mtime_ns,
                int(active),
            ),
        )
        connection.commit()


class BeforeTrainingSampleInspectionTests(unittest.TestCase):
    def test_already_matched_manifest_item_skips_raw_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = _settings(root)
            settings.file_management.export_dir.mkdir(parents=True)
            settings.file_management.video_root.mkdir(parents=True)
            osz_path = settings.file_management.export_dir / "song.osz"
            _write_osz(osz_path)
            folder_name = "item_0001"
            folder = settings.file_management.target_root / folder_name
            folder.mkdir(parents=True)
            (folder / "item_0001.mp4").write_bytes(b"already matched video")
            _write_manifest_db(
                settings.file_management.target_root,
                folder_name=folder_name,
                source_name="Song Normal",
                source_osz_name=osz_path.name,
                source_mtime_ns=osz_path.stat().st_mtime_ns,
            )

            report = inspect_before_training_samples(
                settings,
                matched_manifest_path=root / "startup_manifest.json",
                run_match_probe=False,
            )

            self.assertEqual(report.raw_candidates_total, 1)
            self.assertEqual(len(report.raw_unmatched_candidates), 0)
            self.assertEqual(len(report.recovered_matched_samples), 1)
            self.assertFalse(report.should_run_before_traning)
            self.assertFalse((settings.file_management.target_root / "manifest.csv").exists())


class StartupFlowTests(unittest.TestCase):
    def test_dry_run_does_not_execute_before_pipeline_or_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = _settings(root)
            folder_name = "item_0002"
            folder = settings.file_management.target_root / folder_name
            folder.mkdir(parents=True)
            (folder / settings.file_management.audio_filename).write_bytes(b"audio")
            settings.file_management.video_root.mkdir(parents=True)
            (settings.file_management.video_root / "candidate.mp4").write_bytes(b"video")
            _write_manifest_db(
                settings.file_management.target_root,
                folder_name=folder_name,
                source_name="Pending Normal",
                source_osz_name="pending.osz",
                source_mtime_ns=123,
            )
            before_startup = StartupCheckReport(
                scope="before_traning.startup_checks",
                results=(
                    StartupCheckResult(
                        key="before_traning:raw_data",
                        status="passed",
                        message="new unmatched raw data can update the training dataset",
                        details={
                            "should_run_before_traning": True,
                            "reason": "test raw data",
                        },
                    ),
                ),
            )
            traning_startup = StartupCheckReport(
                scope="traning.startup_checks",
                results=(
                    StartupCheckResult(
                        key="traning:data_input",
                        status="passed",
                        message="ok",
                    ),
                ),
            )
            split_sync = DatasetSplitSyncResult(
                manifest_path=root / "dataset_split_manifest.json",
                created=True,
                changed=True,
                dry_run=True,
                new_items=(
                    DatasetSplitItem(
                        item_name="item_0002",
                        split="train",
                        segment_count=1,
                        assigned_at_utc="2026-06-24T00:00:00+00:00",
                        assignment_reason="test",
                    ),
                ),
                manifest=DatasetSplitManifest(
                    seed=2026,
                    ratios=SplitRatios(),
                    items={},
                ),
            )

            with (
                patch("start.flow.load_before_settings", return_value=settings),
                patch("start.flow.load_training_settings", return_value=object()),
                patch("start.flow._sync_dataset_splits", return_value=split_sync),
                patch(
                    "start.flow.run_before_startup_checks",
                    return_value=before_startup,
                ),
                patch(
                    "start.flow.run_traning_startup_checks",
                    return_value=traning_startup,
                ),
                patch("start.flow.TRAINING_PIPELINE.run_direct") as before_run,
                patch("start.flow.run_full_training_pipeline") as full_training,
            ):
                result = run_startup_flow(
                    StartupFlowConfig(
                        training_config=Path("unused.yaml"),
                        device=torch.device("cpu"),
                        matched_manifest_path=root / "startup_manifest.json",
                        before_match_probe=False,
                        test_level="none",
                        dry_run=True,
                    )
                )

            self.assertTrue(result.before_startup.ok)
            self.assertEqual(result.before_run.status, "skipped")
            before_run.assert_not_called()
            full_training.assert_not_called()


if __name__ == "__main__":
    unittest.main()
