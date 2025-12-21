"""Configuration constants for data transformation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    TRAIN_SIZE: float = 0.81
    VAL_SIZE: float = 0.81

    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
