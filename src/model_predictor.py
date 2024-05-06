import torch

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from data_transformer import DataTransformation
from utils.unet_parts.unet_model import UNET
from utils.constants import constants


class ModelPrediction:
    def __init__(self, model_path) -> None:
        _, self.transform = DataTransformation(None, None).get_transformation_pipelines()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = UNET(in_channels=3, out_channels=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
    
    def return_mask_array(self, image_path: str) -> np.array:
        '''
        A function that returns mask prediction array.

        Arguments:
            image_path: str

        Returns:
            np.array (numpy)
        '''
        image = np.array(Image.open(image_path).convert('RGB'))
        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        
        self.model.eval()

        with torch.no_grad():
            prediction = self.model(image)
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > constants['sigmoid_threshold']).float()
            prediction = prediction.squeeze()
            prediction = prediction.cpu()
            prediction = prediction.numpy()
        
        return prediction * 255
    
    def create_comparison_image(self, image_path: str, mask: np.array) -> None:
        '''
        A function that creates a comparison image.

        Arguments:
            image_path: str
            mask: np.array (numpy)
        '''
        predicted_image = Image.open(image_path)
        predicted_mask = Image.fromarray(mask).convert('L')

        original_image = Image.open(image_path)
        original_mask = Image.open(image_path.replace('images', 'masked_images'))

        predicted_image.paste(predicted_mask, (0, 0), mask=predicted_mask)
        original_image.paste(original_mask, (0, 0), mask=original_mask)

        plt.subplots(1, 2, figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.axis('off')
        plt.title('Ground Truth')
        
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_image)
        plt.axis('off')
        plt.title('Predicted mask')

        plt.show()
