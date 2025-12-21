from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from tqdm import tqdm

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchinfo import summary

from .constants import Constants
from .utils import plot_history, save_model


class ModelTraining:
    def __init__(
        self, model: Module, loss_fn: Module, optimizer: Optimizer, experiment_name: str
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            loss_fn: Loss function
            optimizer: Optimizer
            experiment_name: Name of current MLflow experiment
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.device = torch.device(Constants.DEVICE)
        self.model.to(self.device)

        mlflow.set_experiment(experiment_name)

    def train_step(self, batch: Tuple) -> float:
        """Executes single training step."""
        self.model.train()

        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate_step(self, batch: Tuple) -> float:
        """Executes single evaluation step."""
        self.model.eval()

        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

        return loss.item()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        plot_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Trains the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            plot_path: Path to save metrics plot

        Returns:
            Training history
        """
        history = {"train_loss": [], "validation_loss": []}

        with mlflow.start_run() as run:
            print(f"MLFlow Run ID: {run.info.run_id}")

            mlflow.log_params(asdict(Constants()))

            for epoch in range(1, Constants.EPOCHS + 1):
                print(f"\n\nEpoch {epoch} / {Constants.EPOCHS}")

                train_losses = []
                train_progress_bar = tqdm(train_loader, desc="Training")

                for batch in train_progress_bar:
                    loss = self.train_step(batch)
                    train_losses.append(loss)

                    train_progress_bar.set_postfix(loss=loss)

                val_losses = []
                val_progress_bar = tqdm(val_loader, desc="Evaluating")

                for batch in val_progress_bar:
                    loss = self.evaluate_step(batch)
                    val_losses.append(loss)

                    val_progress_bar.set_postfix(loss=loss)

                train_loss = sum(train_losses) / len(train_losses)
                val_loss = sum(val_losses) / len(val_losses)

                history["train_loss"].append(train_loss)
                history["validation_loss"].append(val_loss)

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                print(
                    f"\n\nTrain loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}"
                )

            mlflow.pytorch.log_model(self.model, "model")

            if plot_path:
                self.plot_history(history, plot_path)
                mlflow.log_artifact(plot_path)

        return history

    def save_model(self, model: Module, model_path: str) -> None:
        """
        Saves model locally.

        Args:
            model: A model that was being used for training
            model_path: A path where the model state dictionary will be saved
        """
        save_model(model, model_path)

    def plot_history(self, history: Dict[str, List[float]], plot_path: str) -> None:
        """
        Plots training history.

        Args:
            history: A dictionary that contains losses for training and validation iterations
            plot_path: A path where the plot image will be saved
        """
        plot_history(history, plot_path)

    def load_constants_as_dictionary() -> Dict[str, Any]:
        """
        Converts constants dataclass to a dictionary:

        Returns:
            Dict[str, Any]: dataclass method to dictionary
        """
        params_dict = asdict(Constants())
        return {key.lower(): value for key, value in params_dict.items()}
