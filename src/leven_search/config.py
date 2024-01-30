from dataclasses import dataclass


@dataclass
class Config:
    max_distance: int | float

    def __init__(self, max_distance: int | float):
        self.max_distance = max_distance
