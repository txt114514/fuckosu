from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from traning.core.model_export import (
    MODEL_ARTIFACT_VERSION,
    ModelArtifactSpec,
    export_model_artifact,
    validate_model_artifact,
)


class ModelExportTests(unittest.TestCase):
    def test_export_model_artifact_copies_files_and_validates_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = root / "settings.yaml"
            temporal = root / "temporal_model.pt"
            metadata = root / "checkpoint_metadata.json"
            settings.write_text("runtime:\n  device: cpu\n", encoding="utf-8")
            temporal.write_bytes(b"temporal-checkpoint")
            metadata.write_text('{"trial_id": "trial-1"}\n', encoding="utf-8")

            result = export_model_artifact(
                ModelArtifactSpec(
                    artifact_id="artifact-smoke",
                    output_dir=root / "artifacts",
                    kind="inference",
                    settings_path=settings,
                    temporal_checkpoint_path=temporal,
                    metadata_path=metadata,
                    score_version="score-v1",
                    candidate_cache_version="cache-v1",
                    code_version="test",
                )
            )

            self.assertEqual(result.version, MODEL_ARTIFACT_VERSION)
            self.assertTrue((result.artifact_dir / "settings.yaml").exists())
            self.assertTrue((result.artifact_dir / "temporal_model.pt").exists())
            self.assertEqual(validate_model_artifact(result.manifest_path), ())


if __name__ == "__main__":
    unittest.main()
