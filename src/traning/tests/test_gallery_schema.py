from __future__ import annotations

import unittest

from pydantic import ValidationError

from traning.state import BatchGalleryRequest


class BatchGalleryRequestTests(unittest.TestCase):
    def test_trials_must_share_score_version(self) -> None:
        with self.assertRaises(ValidationError):
            BatchGalleryRequest.model_validate(
                {
                    "batch_id": "mixed",
                    "trials": [
                        {
                            "trial_id": "a",
                            "score": 0.9,
                            "score_version": "point-slider-v2+macro-v1",
                            "parameters": {},
                        },
                        {
                            "trial_id": "b",
                            "score": 0.8,
                            "score_version": "external",
                            "parameters": {},
                        },
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
