import os
import re

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

        medical_df = pd.DataFrame({'image_path': images_list, 'masked_image_path': masks_list})
        medical_df['tumor'] = medical_df.masked_image_path.map(self.map_diagnose)

        return medical_df

    def split_data(self, train_size: float, validation_size: float, output_directory: str) -> None:
        '''
        A function that splits all images into train, validation and test categories.

        Arguments:
            train_size: float
            validation_size: float
            output_directory: str
        '''
        medical_df = self.create_dataset()

        train_df, remaining = train_test_split(medical_df,
                                               train_size=train_size,
                                               stratify=medical_df.tumor)
        
        val_df, test_df = train_test_split(remaining,
                                           test_size=validation_size,
                                           stratify=remaining.tumor)
        
        del remaining

        train_df.to_csv(os.path.join(output_directory, 'train_df.csv'), index=False)
        val_df.to_csv(os.path.join(output_directory, 'val_df.csv'), index=False)
        test_df.to_csv(os.path.join(output_directory, 'test_df.csv'), index=False)
    
    def get_transformation_pipelines(self) -> Tuple[transforms.Compose, transforms.Compose]:
        '''
        A function that returns two transformation pipelines, each for different task.
        For training and for validation (test dataset uses val_transform_pipeline).

        Returns:
            Tuple[A.Compose, A.Compose]
        '''                
        train_transform_pipeline = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((256, 256), antialias=True)])
        
        val_test_transform_pipeline = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Resize((256, 256), antialias=True)])
        
        return train_transform_pipeline, val_test_transform_pipeline
    
    def get_custom_datasets(self, data_directory) -> Tuple[BrainMRIDataset, BrainMRIDataset, BrainMRIDataset]:
        '''
        A function that returns three custom datasets.
        One for training, one for validation and one for testing.

        Arguments:
            data_directory: str

        Returns:
            Tuple[BrainMRIDataset, BrainMRIDataset, BrainMRIDataset]
        '''
        train_transform_pipeline, val_test_transform_pipeline = self.get_transformation_pipelines()

        train_df = pd.read_csv(os.path.join(data_directory, 'train_df.csv'))
        val_df = pd.read_csv(os.path.join(data_directory, 'train_df.csv'))
        test_df = pd.read_csv(os.path.join(data_directory, 'train_df.csv'))

        train_dataset = BrainMRIDataset(train_df,
                                        self.images_path,
                                        self.masked_images_path,
                                        train_transform_pipeline)
        
        val_dataset = BrainMRIDataset(val_df,
                                      self.images_path,
                                      self.masked_images_path,
                                      val_test_transform_pipeline)
        
        test_dataset = BrainMRIDataset(test_df,
                                       self.images_path,
                                       self.masked_images_path,
                                       val_test_transform_pipeline)
        
        return train_dataset, val_dataset, test_dataset

    
    def get_data_loaders(self, data_directory: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        '''
        A function that returns data loaders for model training.
        One for training, one for validation and one for testing.

        Arguments:
            data_directory: str

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]
        '''
        train_dataset, val_dataset, test_dataset = self.get_custom_datasets(data_directory)

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
        
        return train_loader, val_loader, test_loader
