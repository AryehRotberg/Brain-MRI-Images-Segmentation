import os

import matplotlib.pyplot as plt

import torch
from torchinfo import summary

from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from tqdm import tqdm

import mlflow

import segmentation_models_pytorch as smp
from src.utils.constants import constants


class ModelTraining:
    def __init__(self, train_loader: DataLoader, validation_loader: DataLoader) -> None:
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = smp.UnetPlusPlus(encoder_name=constants['encoder_name'],
                                      encoder_weights=constants['encoder_weights'],
                                      in_channels=3,
                                      classes=1).to(self.device)
        
        self.loss_fn = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=constants['learning_rate'])

        mlflow.set_experiment('Brain MRI Medical Images Segmentation')
    
    def _train_step(self) -> float:
        '''
        Executes one training step over the entire dataset.

        Returns:
            train_loss: Average training loss for the epoch
        '''
        self.model.train()
        train_loss = 0
        progress_bar = tqdm(self.train_loader, desc='Training')

        for images, masked_images in progress_bar:
            images = images.to(self.device)
            masked_images = masked_images.to(self.device)

            prediction = self.model(images)

            loss = self.loss_fn(prediction, masked_images)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
        
        train_loss = train_loss / len(self.train_loader)
        return train_loss
    
    def _evaluate_step(self) -> float:
        '''
        Executes one evaluation step over the validation dataset.

        Returns:
            validation_loss: Average validation loss for the epoch
        '''
        self.model.eval()
        validation_loss = 0
        progress_bar = tqdm(self.validation_loader, desc='Evaluating')

        with torch.no_grad():
            for images, masked_images in progress_bar:
                images = images.to(self.device)
                masked_images = masked_images.to(self.device)
                
                prediction = self.model(images)

                loss = self.loss_fn(prediction, masked_images)
                validation_loss += loss.item()

                progress_bar.set_postfix(loss=loss.item())
        
        validation_loss = validation_loss / len(self.validation_loader)
        return validation_loss
    
    def train(self, verbose: bool=True, plot_output_path: str = None) -> dict:
        '''
        Executes the training loop for multiple epochs.

        Arguments:
            verbose: Whether to print the training progress
            plot_output_path: Path to save the loss plot (optional)

        Returns:
            history: Dictionary containing train and validation losses
        '''
        history = {'train_loss': [], 'validation_loss': []}

        with mlflow.start_run() as run:
            if verbose:
                print(f'MLFlow Run ID: {run.info.run_id}')
            
            mlflow.log_params(constants)
            
            for epoch in range(constants['epochs']):
                if verbose:
                    print(f'\n\nEpoch {epoch + 1} / {constants["epochs"]}')

                train_loss = self._train_step()
                val_loss = self._evaluate_step()

                history['train_loss'].append(train_loss)
                history['validation_loss'].append(val_loss)
                
                if verbose:
                    print(f'\n\nTrain loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}')
            
            mlflow.pytorch.log_model(self.model, 'model')

            if plot_output_path is not None:
                self._plot_history(history, plot_output_path)
                mlflow.log_artifact(plot_output_path)
        
        return history
    
    def save_model(self, output_directory: str) -> None:
        '''
        A function that saves model state dictionary for later usage.

        Arguments:
            output_directory: Directory where the model will be saved
        '''
        torch.save(self.model.state_dict(), os.path.join(output_directory, 'model.pth'))
    
    def get_model_summary(self) -> None:
        summary(self.model, input_data=next(iter(self.train_loader))[0].to(self.device))
    
    @staticmethod
    def _plot_history(history: dict, output_path: str):
        '''
        A function that plots the loss graph for both training and validation datasets.

        Arguments:
            history: Dictionary containing loss values
            output_path: Path to save the plot
        '''
        plt.plot(history['train_loss'])
        plt.plot(history['validation_loss'])
        plt.legend(['Train Loss', 'Validation Loss'])
        plt.savefig(output_path)
    