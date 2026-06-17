from __future__ import annotations

from collections.abc import Mapping
import importlib
import sys


LEGACY_MODULE_ALIASES: Mapping[str, str] = {
    "traning.core.data_input": "traning.core.dataset_import",
    "traning.core.data_input.data_input": "traning.core.dataset_import.data_input",
    "traning.core.data_input.loader": "traning.core.dataset_import.loader",
    "traning.core.data_input.pipeline": "traning.core.dataset_import.pipeline",
    "traning.core.data_input.preflight": "traning.core.dataset_import.preflight",
    "traning.core.env_check": "environment.env_check",
    "traning.core.memory": "traning.Lib.runtime.memory",
    "traning.data": "traning.Lib.data",
    "traning.data.color_cues": "traning.Lib.data.color_cues",
    "traning.data.coordinates": "traning.Lib.data.coordinates",
    "traning.data.patch_stream": "traning.Lib.data.patch_stream",
    "traning.data.synthetic_structures": "traning.Lib.data.synthetic_structures",
    "traning.models": "traning.Lib.models",
    "traning.models.gated_sparse_fusion": "traning.Lib.models.gated_sparse_fusion",
    "traning.models.global_encoder": "traning.Lib.models.global_encoder",
    "traning.models.global_structure_head": (
        "traning.Lib.models.global_structure_head"
    ),
    "traning.models.local_encoder": "traning.Lib.models.local_encoder",
    "traning.models.object_heads": "traning.Lib.models.object_heads",
    "traning.models.outputs": "traning.Lib.models.outputs",
    "traning.models.stack": "traning.Lib.models.stack",
    "traning.models.temporal_model": "traning.Lib.models.temporal_model",
    "traning.training": "traning.Lib.compat.training",
    "traning.training.feature_canvas": "traning.Lib.training.feature_canvas",
    "traning.training.losses": "traning.Lib.training.losses",
    "traning.training.spatial_decode": "traning.Lib.training.spatial_decode",
    "traning.training.spatial_inference": (
        "traning.core.spatial_training.spatial_inference"
    ),
    "traning.training.spatial_targets": "traning.Lib.training.spatial_targets",
    "traning.training.spatial_trainer": (
        "traning.core.spatial_training.spatial_trainer"
    ),
}


def install_legacy_training_aliases(
    aliases: Mapping[str, str] = LEGACY_MODULE_ALIASES,
) -> None:
    for legacy_name, target_name in aliases.items():
        module = sys.modules.get(legacy_name)
        if module is None:
            module = importlib.import_module(target_name)
            sys.modules[legacy_name] = module
        _bind_parent_attribute(legacy_name, module)


def _bind_parent_attribute(module_name: str, module: object) -> None:
    parent_name, _, attribute = module_name.rpartition(".")
    if not parent_name:
        return
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, attribute, module)
