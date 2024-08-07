import argparse

from src.components.model_predictor import ModelPrediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A prediction pipeline for brain MRI segmentation.')
    parser.add_argument('--model_path', type=str, help='A path to a trained model.')
    parser.add_argument('--image_path', type=str, help='A path to an image to be segmented.')
    args = parser.parse_args()

    model_predictor = ModelPrediction(args.model_path)
    model_predictor.create_comparison_image(args.image_path, model_predictor.get_mask_pred_array(args.image_path))
