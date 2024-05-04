import os
import shutil
import re

from tqdm import tqdm

from logger import logger


class DataExtraction:
    def __init__(self, raw_data_dir: str, images_dir: str, masked_images_dir: str) -> None:
        self.raw_data_dir = raw_data_dir
        self.images_dir = images_dir
        self.masked_images_dir = masked_images_dir
    
    def move_images_to_directories(self) -> None:
        '''
        A function that moves images from raw directory to
        corresponding directories -> "images" and "masked_images".
        '''
        dirs = [_[0] for _ in os.walk(self.raw_data_dir)][1:]

        for patient_path in tqdm(dirs):
            images = os.listdir(patient_path)
            images.sort(key=lambda _ : int(re.sub('\D', '', _)))
            
            for image_path in images:
                if '_mask' not in image_path:
                    shutil.copy(os.path.join(patient_path, image_path), os.path.join(self.images_dir, image_path))
                else:
                    shutil.copy(os.path.join(patient_path, image_path), os.path.join(self.masked_images_dir, image_path))
        
        logger.info('Moved images to corresponding directories.')
    
    def rename_images_by_index(self) -> None:
        '''
        A function that renames images by index.
        '''
        images_list = os.listdir(self.images_dir)
        masked_images_list = os.listdir(self.masked_images_dir)

        images_list.sort(key=lambda _ : int(re.sub('\D', '', _)))
        masked_images_list.sort(key=lambda _ : int(re.sub('\D', '', _)))

        for idx, image_path in tqdm(enumerate(images_list)):
            os.replace(os.path.join(self.images_dir, image_path),
                       os.path.join(self.images_dir, f'image_{idx}.tif'))

        for idx, image_path in tqdm(enumerate(masked_images_list)):
            os.replace(os.path.join(self.masked_images_dir, image_path),
                       os.path.join(self.masked_images_dir, f'image_{idx}.tif'))
        
        logger.info('Renamed images by index.')
