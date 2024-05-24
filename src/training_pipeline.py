import argparse

from logging_handler import Logger

from data_extractor import DataExtraction
from data_transformer import DataTransformation
from model_trainer import ModelTraining
from utils.constants import constants


if __name__ == '__main__':
    logger = Logger('training_pipeline.log').create_logger()
    
    parser = argparse.ArgumentParser(description='A training pipeline for brain MRI segmentation.')
    parser.add_argument('--raw_data_path', type=str, help='A path to a directory that contains raw data.')
    parser.add_argument('--images_path', type=str, help='A directory that contains images only.')
    parser.add_argument('--masked_images_path', type=str, help='A directory that contains masked images only.')
    parser.add_argument('--model_path', type=str, help='A path to a trained model.')
    parser.add_argument('--sorted_data_available', type=bool, help='A boolean value that indicates if the data is already sorted ("" stands for False statement).')
    args = parser.parse_args()

    # Data Extraction
    if not args.sorted_data_available:
        de = DataExtraction(args.raw_data_path, args.images_path, args.masked_images_path)
        de.move_images_to_directories()
        logger.info('Moved images to corresponding directories.')

        de.rename_images_by_index()
        logger.info('Renamed images by index.')

    # Data Transformation
    dt = DataTransformation(args.images_path, args.masked_images_path)
    dt.split_data(train_size=constants['train_size'],
                  validation_size=constants['validation_size'],
                  output_directory='outputs/data')
    
    logger.info('Splitted data into train/val/test categories.')
    
    train_loader, val_loader, test_loader = dt.get_data_loaders(data_directory='outputs/data')
    logger.info('Created 3 data loaders for training, validation and testing.')

    # Model Training
    mt = ModelTraining(train_loader, val_loader)
    mt.train(plot_output_path='outputs/history.png')
    mt.save_model('models')

    logger.info('Trained a model and saved it to models directory.')
