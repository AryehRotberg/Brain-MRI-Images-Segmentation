import os

import matplotlib.pyplot as plt

import torch

from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from tqdm import tqdm

import mlflow

import segmentation_models_pytorch as smp
from src.utils.constants import constants


class ModelTraining:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = smp.UnetPlusPlus(encoder_name=constants['encoder_name'],
                                      encoder_weights=constants['encoder_weights'],
                                      in_channels=3,
                                      classes=1).to(self.device)
        
        self.loss_fn = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=constants['learning_rate'])

        mlflow.set_experiment('Brain MRI Medical Images Segmentation')
    
    def train_step(self) -> float:
        '''
        A function that executes train step.

        Returns:
            train_loss: float
        '''
        self.model.train()

        loop = tqdm(self.train_loader)

        train_loss = 0

        for images, masked_images in loop:
            images = images.to(self.device)
            masked_images = masked_images.to(self.device)

            prediction = self.model(images)
            loss = self.loss_fn(prediction, masked_images)
            train_loss += loss.item()

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            loop.set_postfix(loss=loss.item())
        
        train_loss = train_loss / len(self.train_loader)
        return train_loss
    
    def evaluate_step(self) -> float:
        '''
        A function that executes evaluation step.

        Returns:
            val_loss: float
        '''
        self.model.eval()

        loop = tqdm(self.val_loader)

        val_loss = 0

        with torch.no_grad():
            for images, masked_images in loop:
                images = images.to(self.device)
                masked_images = masked_images.to(self.device)
                
                prediction = self.model(images)
                loss = self.loss_fn(prediction, masked_images)
                val_loss += loss.item()

                loop.set_postfix(loss=loss.item())
        
        val_loss = val_loss / len(self.val_loader)
        return val_loss
    
    def train(self, verbose: bool=True, plot_output_path: str = None) -> dict:
        '''
        A function that executes an entire training epoch.

        Arguments:
            verbose: bool
            plot_output_path: str

        Returns:
            history: dict
        '''
        history = {
            'train_loss': [],
            'val_loss': []
        }

        with mlflow.start_run() as run:
            if verbose:
                print(f'MLFlow Run ID: {run.info.run_id}')
            
            mlflow.log_params(constants)
            
            for epoch in range(constants['epochs']):
                if verbose:
                    print(f'\n\nEpoch {epoch + 1} / {constants["epochs"]}')

                train_loss = self.train_step()
                val_loss = self.evaluate_step()
                
                if verbose:
                    print(f'\n\nTrain loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}')
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
            
            mlflow.pytorch.log_model(self.model, 'model')

            if plot_output_path is not None:
                self.plot_history(history, plot_output_path)
                mlflow.log_artifact(plot_output_path)
        
        return history
    
    def plot_history(self, history: dict, output_path: str):
        '''
        A function that plots the loss graph for both training and validation datasets.

        Arguments:
            history: dict
            output_path: str
        '''
        plt.plot(history['train_loss'])
        plt.plot(history['val_loss'])
        plt.legend(['Train Loss', 'Validation Loss'])

        plt.savefig(output_path)
    
    def save_model(self, output_directory: str) -> None:
        '''
        A function that saves model state dictionary for later usage.

        Arguments:
            output_directory: str
        '''
        torch.save(self.model.state_dict(), os.path.join(output_directory, 'model.pth'))
