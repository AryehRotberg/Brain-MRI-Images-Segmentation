import os

import numpy as np
import pandas as pd

import torch
import numpy as np
from PIL import Image

from torchvision import transforms


class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 images_path: str,
                 masked_images_path: str,
                 transform: transforms.Compose=None) -> None:
        
        self.images_path = images_path
        self.masked_images_path = masked_images_path
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, index):
        image = self.dataframe.iloc[index].image_path
        mask = self.dataframe.iloc[index].masked_image_path
        
        image_array = np.array(Image.open(os.path.join(self.images_path, image)).convert('RGB'))
        mask_array = np.array(Image.open(os.path.join(self.masked_images_path, mask)).convert('L'))

        if self.transform is not None:
            image_array = self.transform(image_array)
            mask_array = self.transform(mask_array)
            
        return (image_array, mask_array)
