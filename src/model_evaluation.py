import os

import numpy as np
import pandas as pd

from tqdm import tqdm

import mlflow

from PIL import Image

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import segmentation_models_pytorch as smp

from model_predictor import ModelPrediction
from utils.constants import constants


class ModelEvaluation:
    def __init__(self,
                 model_path: str,
                 images_path: str,
                 masked_images_path: str,
                 val_df: pd.DataFrame,
                 mlflow_run_id: str = None) -> None:
        
        self.images_path = images_path
        self.masked_images_path = masked_images_path
        self.val_df = val_df
        self.mlflow_run_id = mlflow_run_id

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = smp.UnetPlusPlus(encoder_name=constants['encoder_name'],
                                      encoder_weights=constants['encoder_weights'],
                                      in_channels=3,
                                      classes=1).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.loss_fn = MSELoss()

        self.get_mask_pred_array = ModelPrediction(model_path).get_mask_pred_array
    
    def get_iou_dataframe(self, output_path: str) -> pd.DataFrame:
        '''
        A function that returns a dataframe of all IOU scores per each image in test loader.

        Arguments:
            output_path: str
        
        Returns:
            pd.DataFrame
        '''
        images_list = self.val_df.image_path.tolist()
        iou_list = []

        for image_path in tqdm(images_list):
            prediction = self.get_mask_pred_array(os.path.join(self.images_path, image_path))
            mask_gt = np.array(Image.open(os.path.join(self.masked_images_path, image_path)).convert('L'))

            iou_list.append(self.calculate_iou(prediction, mask_gt))
        
        iou_df = pd.DataFrame({'image_path': images_list, 'tumor': self.val_df.tumor, 'iou': iou_list})
        iou_df.to_csv(output_path, index=False)
        
        if self.mlflow_run_id is not None:
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_artifact(output_path)

        return iou_df
    
    def get_confusion_matrix(self, csv_path: str, plot_path: str):
        '''
        A function that creates a confusion matrix for the input data loader.

        Arguments:
            csv_path: str
            plot_path: str
        
        Returns:
            pd.DataFrame
        '''
        images_list = self.val_df.image_path.tolist()
        predictions = []

        for image_path in tqdm(images_list):
            prediction = self.get_mask_pred_array(os.path.join(self.images_path, image_path))
            predictions.append(prediction.max() / 255)
        
        matrix = confusion_matrix(self.val_df.tumor, predictions)
        matrix_df = pd.DataFrame(matrix)
        matrix_df.to_csv(csv_path, index=True)

        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=None)
        disp.plot().figure_.savefig(plot_path)
        
        if self.mlflow_run_id is not None:
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_artifact(csv_path)
                mlflow.log_artifact(plot_path)
        
        return matrix

    def get_classification_report(self, output_path: str):
        '''
        A function that creates a classification report for the input data loader.

        Arguments:
            output_path: str
        
        Returns:
            pd.DataFrame
        '''
        images_list = self.val_df.image_path.tolist()
        predictions = []

        for image_path in tqdm(images_list):
            prediction = self.get_mask_pred_array(os.path.join(self.images_path, image_path))
            predictions.append(prediction.max() / 255)
        
        report = pd.DataFrame(classification_report(self.val_df.tumor, predictions, output_dict=True)).round(2).transpose()
        report.to_csv(output_path, index=True)

        if self.mlflow_run_id is not None:
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_artifact(output_path)
        
        return report

    def calculate_loss(self, val_loader: DataLoader) -> float:
        '''
        A function that returns loss of a loader.

        Arguments:
            val_loader: DataLoader

        Returns:
            float
        '''
        self.model.eval()

        loop = tqdm(val_loader)

        test_loss = 0

        with torch.no_grad():
            for images, masked_images in loop:
                images = images.to(self.device)
                masked_images = masked_images.to(self.device)
                
                prediction = self.model(images)
                loss = self.loss_fn(prediction, masked_images)
                test_loss += loss.item()

                loop.set_postfix(loss=loss.item())

        val_loss = test_loss / len(val_loader)

        return val_loss
    
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
