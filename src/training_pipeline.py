import argparse

from data_extractor import DataExtraction
from data_transformer import DataTransformation
from model_trainer import ModelTraining
from model_evaluation import ModelEvaluation
from utils.constants import constants


if __name__ == '__main__':
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
        de.rename_images_by_index()

    # Data Transformation
    dt = DataTransformation(args.images_path, args.masked_images_path)
    dt.split_data(train_size=constants['train_size'], validation_size=constants['validation_size'])
    train_loader, val_loader, test_loader = dt.get_data_loaders()

    # Model Training
    mt = ModelTraining(train_loader, val_loader)
    history, mlflow_run_id = mt.train(plot_output_path='outputs/history.png')
    mt.save_predictions_as_images('outputs')
    mt.save_model('models')

    # Model Evaluation
    me = ModelEvaluation(args.model_path, mlflow_run_id)

    loss = me.calculate_loss(test_loader)
    
    iou_df = me.get_iou_dataframe(args.images_path,
                                  args.masked_images_path,
                                  dt.test_df,
                                  'outputs/iou_df.csv')
    
    me.get_confusion_matrix(args.images_path,
                            dt.test_df,
                            'outputs/confusion_matrix.csv')
    
    me.get_classification_report(args.images_path,
                                 dt.test_df,
                                 'outputs/classification_report.csv')
