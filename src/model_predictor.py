import torch

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import cv2

from torchvision import transforms

import segmentation_models_pytorch as smp
from utils.constants import constants


class ModelPrediction:
    def __init__(self, model_path) -> None:
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((256, 256), antialias=True)])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = smp.UnetPlusPlus(encoder_name=constants['encoder_name'],
                                      encoder_weights=constants['encoder_weights'],
                                      in_channels=3,
                                      classes=1).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    
    def get_mask_pred_array(self, image_path: str) -> np.array:
        '''
        A function that returns predicted mask as a numpy array.

        Arguments:
            image_path: str

        Returns:
            np.array (numpy)
        '''
        array_image = np.array(Image.open(image_path).convert('RGB'))
        transformed_image = self.transform(array_image)
        transformed_image = transformed_image.unsqueeze(0)
        transformed_image = transformed_image.to(self.device)
                
        self.model.eval()

        with torch.no_grad():
            prediction = self.model(transformed_image)
        
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > constants['sigmoid_threshold']).float()
        prediction = prediction.squeeze()
        prediction = prediction.cpu()
        prediction = prediction.numpy()
        prediction = prediction * 255
        
        return prediction
    
    def draw_bounding_boxes(self, mask: np.array) -> np.array:
        '''
        Arguments:
            mask: np.array (numpy)
        
        Returns:
            np.array (numpy)
        '''
        mask = mask.astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), 1)
        
        return mask
    
    def create_comparison_image(self, image_path: str, mask: np.array) -> None:
        '''
        A function that creates a comparison image.

        Arguments:
            image_path: str
            mask: np.array (numpy)
        '''
        predicted_image = Image.open(image_path)
        predicted_mask = Image.fromarray(self.draw_bounding_boxes(mask)).convert('L')

        original_image = Image.open(image_path)
        original_mask = np.array(Image.open(image_path.replace('images', 'masked_images')))
        original_mask_with_bbox = self.draw_bounding_boxes(original_mask)
        original_mask_with_bbox = Image.fromarray(original_mask_with_bbox).convert('L')

        predicted_image.paste(predicted_mask, (0, 0), mask=predicted_mask)
        original_image.paste(original_mask_with_bbox, (0, 0), mask=original_mask_with_bbox)

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
