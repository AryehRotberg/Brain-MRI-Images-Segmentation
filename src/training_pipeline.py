import argparse

import matplotlib.pyplot as plt

from data_extractor import DataExtraction
from data_transformer import DataTransformation
from model_trainer import ModelTraining


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A training pipeline for brain MRI segmentation.')
    parser.add_argument('--raw_data_path', type=str, help='A path to a directory that contains raw data.')
    parser.add_argument('--images_path', type=str, help='A directory that contains images only.')
    parser.add_argument('--masked_images_path', type=str, help='A directory that contains masked images only.')
    parser.add_argument('--sorted_data_available', type=bool, help='A boolean value that indicates if the data is already sorted ("" stands for False statement).')
    args = parser.parse_args()

    # Data Extraction
    if not args.sorted_data_available:
        de = DataExtraction(args.raw_data_path, args.images_path, args.masked_images_path)
        de.move_images_to_directories()
        de.rename_images_by_index()

    # Data Transformation
    dt = DataTransformation(args.images_path, args.masked_images_path)
    dt.split_data(train_size=0.81, validation_size=0.12)
    train_loader, val_loader, test_loader = dt.get_data_loaders()

    # Model Training
    mt = ModelTraining(train_loader, val_loader)
    history = mt.train()
    mt.save_predictions_as_imgs('outputs')
    mt.save_model('models')

    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.show()
