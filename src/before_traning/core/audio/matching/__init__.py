"""公开基于音频内容匹配录像的组装入口与兼容构建器。"""

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
