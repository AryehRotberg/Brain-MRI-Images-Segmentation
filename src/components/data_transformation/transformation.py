import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.datasets.custom_dataset import BrainMRIDataset
from .constants import Constants
from .utils import save_as_csv


class DataTransformation:
    """Handles data transformation and preparation for model training."""

    def map_diagnose(self, target_image_path: str, target_images_dir: str) -> int:
        """
        Maps diagnosis label based on mask presence

        Args:
            image_path (str): Path where the mask image is stored.
            target_images_dir (str): Directory where all target images are stored.

        Returns:
            int: 1 if a mask is present, 0 if otherwise.
        """
        image_array = np.array(
            Image.open(os.path.join(target_images_dir, target_image_path)).convert("L")
        )
        return 1 if image_array.max() > 0 else 0

    def create_medical_dataframe(
        self, input_images_dir: str, target_images_dir: str
    ) -> pd.DataFrame:
        """
        Creates a Pandas dataframe from image directories.

        Args:
            input_images_dir (str): Directory where input images are stored.
            target_images_dir (str): Directory where target images are stored.

        Returns:
            pd.DataFrame: Pandas dataframe with input and target images.
        """
        tqdm.pandas(desc="Mapping diagnosis to target images")

        images_list = os.listdir(input_images_dir)
        masks_list = os.listdir(target_images_dir)

        images_list.sort(key=lambda _: int(re.sub("\D", "", _)))
        masks_list.sort(key=lambda _: int(re.sub("\D", "", _)))

        medical_df = pd.DataFrame(
            {"input_image_path": images_list, "target_image_path": masks_list}
        )
        medical_df["tumor"] = medical_df.target_image_path.progress_apply(
            lambda x: self.map_diagnose(x, target_images_dir)
        )

        return medical_df

    def split_data(
        self, dataframe: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits Pandas medical dataframe into train, validation and test sets.

        Args:
            dataframe (pd.DataFrame): Pandas dataframe that includes all input and target images.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Splitted data into three sections.
        """
        train_df, temp_df = train_test_split(
            dataframe, train_size=Constants.TRAIN_SIZE, stratify=dataframe.tumor
        )

        val_df, test_df = train_test_split(
            temp_df, test_size=Constants.VAL_SIZE, stratify=temp_df.tumor
        )

        del temp_df

        return train_df, val_df, test_df

    def create_dataset(
        self, dataframe: pd.DataFrame, input_images_dir: str, target_images_dir: str
    ) -> BrainMRIDataset:
        """
        Creates dataset from a Pandas dataframe.

        Args:
            dataframe (pd.DataFrame): Pandas dataframe that is one of the three sections: train, val or test.

        Returns:
            Custom PyTorch dataset.
        """
        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((256, 256), antialias=True)]
        )

        return BrainMRIDataset(
            dataframe, input_images_dir, target_images_dir, transformation
        )

    def create_data_loader(self, dataset: BrainMRIDataset, shuffle: bool) -> DataLoader:
        """
        Creates DataLoader object for model training.

        Args:
            dataset (BrainMRIDataset): A custom PyTorch dataset.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: A PyTorch DataLoader configured with the provided dataset and settings.
        """
        return DataLoader(
            dataset,
            batch_size=Constants.BATCH_SIZE,
            num_workers=Constants.NUM_WORKERS,
            pin_memory=Constants.PIN_MEMORY,
            shuffle=shuffle,
        )

    def save_as_csv(self, dataframe: pd.DataFrame, output_path: str) -> None:
        """
        Saves Pandas DataFrame to a csv file, excluding index column.

        Args:
            dataframe (pd.DataFrame): DataFrame to be saved as a CSV file.
            output_path (str): File path where the CSV file will be saved.
        """
        save_as_csv(dataframe, output_path)
