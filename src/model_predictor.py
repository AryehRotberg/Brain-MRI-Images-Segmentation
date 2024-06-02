import torch

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image, ImageChops

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

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(mask_bgr, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        return mask_bgr
    
    def create_comparison_image(self, image_path: str, mask: np.array) -> None:
        '''
        A function that creates a comparison image.

        Arguments:
            image_path: str
            mask: np.array (numpy)
        '''
        predicted_image = Image.open(image_path).convert('RGBA')
        predicted_mask_with_bbox = Image.fromarray(self.draw_bounding_boxes(mask)).convert('RGBA')

        original_image = Image.open(image_path).convert('RGBA')
        original_mask = np.array(Image.open(image_path.replace('images', 'masked_images')))
        original_mask_with_bbox = Image.fromarray(self.draw_bounding_boxes(original_mask)).convert('RGBA')

        result_predicted = ImageChops.screen(predicted_image, predicted_mask_with_bbox)
        result_original = ImageChops.screen(original_image, original_mask_with_bbox)

        plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle(f'Intersection over Union (IOU): {self.calculate_iou(mask, original_mask)}')

        plt.subplot(1, 2, 1)
        plt.imshow(result_original)
        plt.axis('off')
        plt.title('Ground Truth')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_predicted)
        plt.axis('off')
        plt.title('Predicted mask')

        plt.show()
    
    def calculate_iou(self, mask_pred, mask_gt, epsilon=1e-5) -> float:
        '''
        A function that returns IOU score of a mask.

        Returns:
            float
        '''
        mask_pred = mask_pred.astype(np.uint8)
        mask_gt = mask_gt.astype(np.uint8)

        mask_pred[mask_pred > 1] = 1
        mask_gt[mask_gt > 1] = 1

        intersection = np.sum(mask_pred * mask_gt)
        union = np.sum(mask_pred + mask_gt - mask_pred * mask_gt)

        iou = round(np.maximum(0, (intersection + epsilon) / (union + epsilon)), 2)
        return iou
