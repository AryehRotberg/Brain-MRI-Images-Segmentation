from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
import torch
from torch.nn import Module
from torchvision import transforms

from src.components.model_training.constants import Constants


class ModelPrediction:
    def __init__(self, model: Module, transform: transforms.Compose) -> None:
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            transform (Optional[transforms.Compose]): Optional PyTorch Vision pipeline to transform input image.
        """
        self.model = model
        self.transform = transform

    def predict_and_transform(self, image_path: str) -> np.ndarray:
        """
        Generates model prediction and transforms output array.

        Args:
            image_path (str): Path where input image is stored.

        Returns:
            np.ndarray: Mask prediction.
        """
        input_array = np.array(Image.open(image_path).convert("RGB"))

        transformed_array = self.transform(input_array)
        transformed_array = transformed_array.unsqueeze(0).to(Constants.DEVICE)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(transformed_array)

        outputs = torch.sigmoid(outputs)
        outputs = (outputs > Constants.SIGMOID_THRESHOLD).float()
        outputs = outputs.squeeze().cpu().numpy() * 255

        return outputs

    def draw_bounding_boxes(self, mask: np.ndarray) -> np.ndarray:
        """
        Draws bounding boxes around detected mask.

        Args:
            mask (np.ndarray): Mask prediction.

        Returns:
            np.ndarray: Mask with drawn bounding boxes.
        """
        mask = mask.astype(np.uint8)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(mask_bgr, (x, y), (x + w, y + h), (255, 0, 0), 1)

        return mask_bgr

    def display_comparison(self, image_path: str, mask: np.ndarray) -> None:
        """
        Displays comparison image between predicted and ground truth masks.

        Args:
            image_path (str): Path where image is stored.
            mask (np.ndarray): Mask prediction.
        """
        predicted_image = Image.open(image_path).convert("RGBA")
        predicted_mask_with_bbox = Image.fromarray(
            self.draw_bounding_boxes(mask)
        ).convert("RGBA")

        original_image = Image.open(image_path).convert("RGBA")
        original_mask = np.array(
            Image.open(image_path.replace("images", "masked_images"))
        )
        original_mask_with_bbox = Image.fromarray(
            self.draw_bounding_boxes(original_mask)
        ).convert("RGBA")

        result_predicted = ImageChops.screen(predicted_image, predicted_mask_with_bbox)
        result_original = ImageChops.screen(original_image, original_mask_with_bbox)

        plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle(
            f"Intersection over Union (IOU): {self.calculate_iou(mask, original_mask)}"
        )

        plt.subplot(1, 2, 1)
        plt.imshow(result_original)
        plt.axis("off")
        plt.title("Ground Truth")

        plt.subplot(1, 2, 2)
        plt.imshow(result_predicted)
        plt.axis("off")
        plt.title("Predicted mask")

        plt.show()

    def calculate_iou(
        self, mask_pred: np.ndarray, mask_gt: np.ndarray, epsilon: float = 1e-5
    ) -> float:
        """
        Calculates Intersection over Union (IoU) between predicted and ground truth masks.

        Args:
            mask_pred (np.ndarray): Predicted mask.
            mask_gt (np.ndarray): Ground truth mask.
            epsilon (float): Constant to avoid division by zero.

        Returns:
            float: IoU score.
        """
        mask_pred = mask_pred.astype(np.uint8)
        mask_gt = mask_gt.astype(np.uint8)

        mask_pred[mask_pred > 1] = 1
        mask_gt[mask_gt > 1] = 1

        intersection = np.sum(mask_pred * mask_gt)
        union = np.sum(mask_pred + mask_gt - mask_pred * mask_gt)

        iou = round(np.maximum(0, (intersection + epsilon) / (union + epsilon)), 2)
        return iou
