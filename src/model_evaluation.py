import os

import numpy as np
import pandas as pd

from tqdm import tqdm

import mlflow

from PIL import Image

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report

from data_transformer import DataTransformation
from utils.unet_parts.unet_model import UNET
from utils.constants import constants


class ModelEvaluation:
    def __init__(self, model_path: str, mlflow_run_id: str = None) -> None:
        self.mlflow_run_id = mlflow_run_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = UNET(in_channels=3, out_channels=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path))

        self.loss_fn = MSELoss()
    
    def get_model_prediction_for_a_single_image(self, image_path: str) -> np.array:
        '''
        A function that returns predicted mask as a numpy array.

        Arguments:
            image_path: str
        
        Returns:
            np.array (numpy)
        '''
        _, transform = DataTransformation(None, None).get_transformation_pipelines()

        image = np.array(Image.open(image_path).convert('RGB'))
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        self.model.eval()

        with torch.no_grad():
            prediction = self.model(image)
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > constants['sigmoid_threshold']).float()
            prediction = prediction.squeeze()
            prediction = prediction.cpu().numpy()
    
        prediction = prediction * 255

        return prediction
    
    def get_iou_dataframe(self,
                          images_path: str,
                          masked_images_path: str,
                          val_df: pd.DataFrame,
                          output_path: str = None) -> pd.DataFrame:
        '''
        A function that returns a dataframe of all IOU scores per each image in test loader.

        Arguments:
            images_path: str
            masked_images_path: str
            val_df: pd.DataFrame
        
        Returns:
            pd.DataFrame
        '''
        images_list = val_df.image_path.tolist()
        iou_list = []

        for image_path in tqdm(images_list):
            prediction = self.get_model_prediction_for_a_single_image(os.path.join(images_path, image_path))
            mask_gt = np.array(Image.open(os.path.join(masked_images_path, image_path)).convert('L'))

            iou_list.append(self.calculate_iou(prediction, mask_gt))
        
        result_df = pd.DataFrame({'image_path': images_list, 'tumor': val_df.tumor, 'iou': iou_list})

        if output_path is not None and self.mlflow_run_id is not None:
            result_df.to_csv(output_path, index=False)
        
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_artifact(output_path)

        return result_df
    
    def get_confusion_matrix(self,
                             images_path: str,
                             val_df: pd.DataFrame,
                             output_path: str = None):
        '''
        A function that creates a confusion matrix for the input data loader.

        Arguments:
            images_path: str
            masked_images_path: str
            val_df: pd.DataFrame
        
        Returns:
            pd.DataFrame
        '''
        images_list = val_df.image_path.tolist()
        predictions = []

        for image_path in tqdm(images_list):
            prediction = self.get_model_prediction_for_a_single_image(os.path.join(images_path, image_path))
            predictions.append(prediction.max() / 255)
        
        matrix = pd.DataFrame(confusion_matrix(val_df.tumor, predictions))

        if output_path is not None and self.mlflow_run_id is not None:
            matrix.to_csv(output_path, index=True)

            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_artifact(output_path)
        
        return matrix

    def get_classification_report(self,
                                  images_path: str,
                                  val_df: pd.DataFrame,
                                  output_path: str = None):
        '''
        A function that creates a classification report for the input data loader.

        Arguments:
            images_path: str
            masked_images_path: str
            val_df: pd.DataFrame
        
        Returns:
            pd.DataFrame
        '''
        images_list = val_df.image_path.tolist()
        predictions = []

        for image_path in tqdm(images_list):
            prediction = self.get_model_prediction_for_a_single_image(os.path.join(images_path, image_path))
            predictions.append(prediction.max() / 255)
        
        report = pd.DataFrame(classification_report(val_df.tumor, predictions, output_dict=True)).round(2).transpose()

        if output_path is not None and self.mlflow_run_id is not None:
            report.to_csv(output_path, index=True)

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
