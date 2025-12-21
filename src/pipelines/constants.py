"""Configuration constants for training pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    RAW_DATA_DIR: str = 'data/raw'
    INPUT_IMAGES_DIR: str = 'data/images'
    TARGET_IMAGES_DIR: str = 'data/masked_images'
    SORTED_DATA_AVAILABLE: bool = True
