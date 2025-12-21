"""Configuration constants for model training."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    ENCODER_NAME: str = 'resnet34'
    ENCODER_WEIGHTS: str = 'imagenet'
    EPOCHS: int = 10
    LEARNING_RATE: float = 0.0003
    SIGMOID_THRESHOLD: float = 0.55
    DEVICE: str = 'cuda'
