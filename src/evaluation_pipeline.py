import argparse

from data_transformer import DataTransformation
from model_evaluation import ModelEvaluation
from utils.constants import constants


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An evaluation pipeline for brain MRI segmentation.')
    parser.add_argument('--images_path', type=str, help='A directory that contains images only.')
    parser.add_argument('--masked_images_path', type=str, help='A directory that contains masked images only.')
    parser.add_argument('--model_path', type=str, help='A path to a trained model.')
    args = parser.parse_args()

    dt = DataTransformation(args.images_path, args.masked_images_path)
    dt.split_data(train_size=constants['train_size'], validation_size=constants['validation_size'])
    train_loader, val_loader, test_loader = dt.get_data_loaders()

    me = ModelEvaluation(args.model_path)
    
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
