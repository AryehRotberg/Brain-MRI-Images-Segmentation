import matplotlib.pyplot as plt

import torch
from torch.nn import Module


def plot_history(history: dict, plot_path: str) -> None:
    """
    Plot training history.

    Args:
        history: A dictionary that contains losses for training and validation iterations
        plot_path: A path where the plot image will be saved
    """
    plt.plot(history['train_loss'])
    plt.plot(history['validation_loss'])
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.savefig(plot_path)

def save_model(model: Module, model_path: str) -> None:
    """
    Save model locally

    Args:
        model: A model that was being used for training
        model_path: A path where the model state dictionary will be saved
    """
    torch.save(model.state_dict(), model_path)
