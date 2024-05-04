import os

import torch
import torchvision

from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils.unet_parts.unet_model import UNET
from utils.dice_loss import DiceLoss
from utils.constants import epochs, learning_rate, sigmoid_threshold


class ModelTraining:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = UNET(in_channels=3, out_channels=1).to(self.device)
        self.loss_fn = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
    
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
    
    def train(self, verbose: bool=True) -> dict:
        '''
        A function that executes an entire training epoch.

        Arguments:
            verbose: bool

        Returns:
            history: dict
        '''
        history = {
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(epochs):
            if verbose:
                print(f'\n\nEpoch {epoch + 1} / {epochs}')

            train_loss = self.train_step()
            val_loss = self.evaluate_step()
            
            if verbose:
                print(f'\n\nTrain loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}')
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
        
        return history
    
    def save_predictions_as_imgs(self, output_directory: str) -> None:
        '''
        A function that saves masked images per batch.

        Arguments:
            output_directory: str
        '''
        self.model.eval()

        for batch_idx, (images, _) in enumerate(self.train_loader):
            images = images.to(device=self.device)

            with torch.no_grad():
                prediction = torch.sigmoid(self.model(images))
                prediction = (prediction > sigmoid_threshold).float()

            torchvision.utils.save_image(prediction, f'{output_directory}/pred_{batch_idx}.png')
    
    def save_model(self, output_directory: str) -> None:
        '''
        A function that saves model state dictionary for later usage.

        Arguments:
            output_directory: str
        '''
        torch.save(self.model.state_dict(), os.path.join(output_directory, 'model.pth'))
