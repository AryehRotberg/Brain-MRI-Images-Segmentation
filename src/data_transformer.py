import os
import re

from logger import logger
from typing import Tuple

import numpy as np
import pandas as pd

from PIL import Image

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from torchvision import transforms

from utils.dataset import BrainMRIDataset
from utils.constants import constants


class DataTransformation:
    def __init__(self, images_path: str, masked_images_path: str) -> None:
        self.images_path = images_path
        self.masked_images_path = masked_images_path
    
    def map_diagnose(self, image_path: str) -> int:
        '''
        A function that maps a diagnose to a given image.

        Arguments:
            image_path: str

        Returns:
            int
        '''
        image_array = np.array(Image.open(os.path.join(self.masked_images_path, image_path)).convert('L'))

        if image_array.max() > 0:
            return 1
        return 0
    
    def create_dataset(self):
        '''
        A function that creates a dataset from images and masks.

        Returns:
            pd.DataFrame
        '''
        images_list = os.listdir(self.images_path)
        masks_list = os.listdir(self.masked_images_path)

        images_list.sort(key=lambda _ : int(re.sub('\D', '', _)))
        masks_list.sort(key=lambda _ : int(re.sub('\D', '', _)))

        self.medical_df = pd.DataFrame({'image_path': images_list, 'masked_image_path': masks_list})
        self.medical_df['tumor'] = self.medical_df.masked_image_path.map(self.map_diagnose)

    def split_data(self, train_size: float, validation_size: float) -> None:
        '''
        A function that splits all images into train, validation and test categories.

        Arguments:
            train_size: float
            validation_size: float
        '''
        self.create_dataset()

        self.train_df, remaining = train_test_split(self.medical_df,
                                                    train_size=train_size,
                                                    stratify=self.medical_df.tumor)
        
        self.val_df, self.test_df = train_test_split(remaining,
                                                     test_size=validation_size,
                                                     stratify=remaining.tumor)
        
        del remaining

        logger.info('Splitted data into train/val/test categories.')
    
    def get_transformation_pipelines(self) -> Tuple[transforms.Compose, transforms.Compose]:
        '''
        A function that returns two transformation pipelines, each for different task.
        For training and for validation (test dataset uses val_transform_pipeline).

        Returns:
            Tuple[A.Compose, A.Compose]
        '''        
        # train_transform_pipeline = transforms.Compose([transforms.ToTensor(),
        #                                                transforms.Resize((256, 256), antialias=True),
        #                                                transforms.RandomHorizontalFlip(),
        #                                                transforms.RandomVerticalFlip(),
        #                                                transforms.RandomRotation(degrees=35),
        #                                                transforms.RandomAffine(degrees=15)])
        
        train_transform_pipeline = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((256, 256), antialias=True)])
        
        val_test_transform_pipeline = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Resize((256, 256), antialias=True)])
        
        return train_transform_pipeline, val_test_transform_pipeline
    
    def get_custom_datasets(self) -> Tuple[BrainMRIDataset, BrainMRIDataset, BrainMRIDataset]:
        '''
        A function that returns three custom datasets.
        One for training, one for validation and one for testing.

        Returns:
            Tuple[BrainMRIDataset, BrainMRIDataset, BrainMRIDataset]
        '''
        train_transform_pipeline, val_test_transform_pipeline = self.get_transformation_pipelines()

        train_dataset = BrainMRIDataset(self.train_df,
                                        self.images_path,
                                        self.masked_images_path,
                                        train_transform_pipeline)
        
        val_dataset = BrainMRIDataset(self.val_df,
                                      self.images_path,
                                      self.masked_images_path,
                                      val_test_transform_pipeline)
        
        test_dataset = BrainMRIDataset(self.test_df,
                                       self.images_path,
                                       self.masked_images_path,
                                       val_test_transform_pipeline)
        
        logger.info('Created 3 custom datasets for training, validation and testing.')
        return train_dataset, val_dataset, test_dataset

    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        '''
        A function that returns data loaders for model training.
        One for training, one for validation and one for testing.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]
        '''
        train_dataset, val_dataset, test_dataset = self.get_custom_datasets()

        train_loader = DataLoader(train_dataset,
                                  batch_size=constants['batch_size'],
                                  num_workers=constants['num_workers'],
                                  pin_memory=True,
                                  shuffle=True)
        
        val_loader = DataLoader(val_dataset,
                                batch_size=constants['batch_size'],
                                num_workers=constants['num_workers'],
                                pin_memory=True,
                                shuffle=False)
        
        test_loader = DataLoader(test_dataset,
                                 batch_size=constants['batch_size'],
                                 num_workers=constants['num_workers'],
                                 pin_memory=True,
                                 shuffle=False)
        
        logger.info('Created 3 data loaders for training, validation and testing.')
        return train_loader, val_loader, test_loader
