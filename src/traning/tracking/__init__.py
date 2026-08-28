"""稳定轨迹身份与确定性候选关联的公开 API。"""

from .association import (
    AssociationCost,
    AssociationCostSpec,
    AssociationCostWeights,
    AssociationMatch,
    AssociationResult,
    GreedyAssociationSolver,
    TrackAssociationView,
    associate_candidates,
)
from .tracker import MultiObjectTracker

__all__ = (
    "AssociationCost",
    "AssociationCostSpec",
    "AssociationCostWeights",
    "AssociationMatch",
    "AssociationResult",
    "GreedyAssociationSolver",
    "MultiObjectTracker",
    "TrackAssociationView",
    "associate_candidates",
)
