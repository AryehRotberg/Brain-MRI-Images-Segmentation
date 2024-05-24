import argparse

from logging_handler import Logger

import pandas as pd

from data_transformer import DataTransformation
from model_evaluation import ModelEvaluation


if __name__ == '__main__':
    logger = Logger('evaluation_pipeline.log').create_logger()

    parser = argparse.ArgumentParser(description='An evaluation pipeline for brain MRI segmentation.')
    parser.add_argument('--images_path', type=str, help='A directory that contains images only.')
    parser.add_argument('--masked_images_path', type=str, help='A directory that contains masked images only.')
    parser.add_argument('--model_path', type=str, help='A path to a trained model.')
    args = parser.parse_args()

    _, _, test_loader = DataTransformation(args.images_path, args.masked_images_path).get_data_loaders(data_directory='outputs/data')

    train_df = pd.read_csv('outputs/data/train_df.csv')
    val_df = pd.read_csv('outputs/data/val_df.csv')
    test_df = pd.read_csv('outputs/data/test_df.csv')

    me = ModelEvaluation(args.model_path,
                         args.images_path,
                         args.masked_images_path,
                         test_df,
                         mlflow_run_id=None)
    
    loss = me.calculate_loss(test_loader)

    logger.info(f'Loss of {test_loader = }: {loss}')

    me.get_iou_dataframe('outputs/iou_df.csv')
    
    logger.info(f'Saved IOU dataframe to outputs directory.')
    
    me.get_confusion_matrix('outputs/confusion_matrix.csv',
                            'outputs/confusion_matrix.png')
    
    logger.info(f'Saved confusion matrix to outputs directory.')
    
    me.get_classification_report('outputs/classification_report.csv')
    
    logger.info(f'Saved classification report to outputs directory.')
