"""Assembly entry for audio-based video matching.

Use:
    from before_traning.core.audio.matching import AudioMatchProcessor
"""

from before_traning.core.audio.matching.matching import (
    AudioMatchExperiment,
    AudioMatchProcessor,
    build_audio_match_experiment_from_config_or_default,
)


__all__ = [
    "AudioMatchExperiment",
    "AudioMatchProcessor",
    "build_audio_match_experiment_from_config_or_default",
]
