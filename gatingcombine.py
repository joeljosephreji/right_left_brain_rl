from enum import Enum, auto

# Enum for specifying the gating combination method
class GatingCombine(Enum):
    SUMMATION = auto()
    SELECT_SAMPLE = auto()

