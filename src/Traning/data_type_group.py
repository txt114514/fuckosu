from dataclasses import dataclass, field
from typing import List, Tuple, Literal

# 限定类型（防止写错字符串）
HitObjectType = Literal["circle", "slider", "spinner"]

@dataclass
class HitObject:
    t_start: int
    t_end: int
    type: HitObjectType = field(init=False)
@dataclass
class Circle(HitObject):
    x: float
    y: float
    def __post_init__(self):
        self.type = "circle"
@dataclass
class Slider(HitObject):
    path: List[Tuple[float, float]]
    repeats: int = 1
    def __post_init__(self):
        self.type = "slider"
@dataclass
class Spinner(HitObject):
    def __post_init__(self):
        self.type = "spinner"