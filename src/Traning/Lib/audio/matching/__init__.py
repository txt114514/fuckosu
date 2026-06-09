"""Assembly entry for audio-based video matching.

Use:
    from Traning.Lib.audio.matching import AudioMatchProcessor
"""

from Traning.Lib.audio.matching.matching import (
    AudioMatchExperiment,
    AudioMatchProcessor,
    build_audio_match_experiment_from_config_or_default,
)


__all__ = [
    "AudioMatchExperiment",
    "AudioMatchProcessor",
    "build_audio_match_experiment_from_config_or_default",
]
