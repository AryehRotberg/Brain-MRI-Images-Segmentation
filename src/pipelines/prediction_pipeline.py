import argparse

import mlflow
import segmentation_models_pytorch as smp
import torch
from torchvision import transforms

from src.components.model_prediction.prediction import ModelPrediction
from src.components.model_training.constants import Constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A prediction pipeline for brain MRI segmentation."
    )
    parser.add_argument("--model_path", type=str, help="A path to a trained model.")
    parser.add_argument(
        "--image_path", type=str, help="A path to an image to be segmented."
    )
    args = parser.parse_args()

    model = smp.UnetPlusPlus(
        encoder_name=Constants.ENCODER_NAME,
        encoder_weights=Constants.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
    ).to(Constants.DEVICE)

    model.load_state_dict(
        torch.load(args.model_path, map_location=Constants.DEVICE, weights_only=True)
    )

    # model = mlflow.pytorch.load_model('runs:/bbc28e10bfc5462db0523212cf58044c/model')

    model_predictor = ModelPrediction(
        model=model,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((256, 256), antialias=True)]
        ),
    )

    model_predictor.display_comparison(
        args.image_path, model_predictor.predict_and_transform(args.image_path)
    )
