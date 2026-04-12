import segmentation_models_pytorch as smp
from torch.nn import MSELoss
from torch.optim import Adam

from src.components.data_ingestion.ingestion import DataIngestion
from src.components.data_transformation.transformation import \
    DataTransformation
from src.components.model_training.constants import \
    Constants as TrainingConstants
from src.components.model_training.training import ModelTraining

from .constants import Constants as PipelineConstants

if __name__ == "__main__":
    # Data Ingestion

    if not PipelineConstants.SORTED_DATA_AVAILABLE:
        data_ingester = DataIngestion()

        data_ingester.move_images_to_directories(
            data_dir=PipelineConstants.RAW_DATA_DIR,
            input_images_dir=PipelineConstants.INPUT_IMAGES_DIR,
            target_images_dir=PipelineConstants.TARGET_IMAGES_DIR,
        )

        data_ingester.rename_images_by_index(
            input_images_dir=PipelineConstants.INPUT_IMAGES_DIR,
            target_images_dir=PipelineConstants.TARGET_IMAGES_DIR,
        )

    # Data Transformation
    data_transformer = DataTransformation()

    medical_df = data_transformer.create_medical_dataframe(
        input_images_dir=PipelineConstants.INPUT_IMAGES_DIR,
        target_images_dir=PipelineConstants.TARGET_IMAGES_DIR,
    )

    train_df, val_df, test_df = data_transformer.split_data(medical_df)

    train_dataset = data_transformer.create_dataset(
        train_df,
        input_images_dir=PipelineConstants.INPUT_IMAGES_DIR,
        target_images_dir=PipelineConstants.TARGET_IMAGES_DIR,
    )

    val_dataset = data_transformer.create_dataset(
        val_df,
        input_images_dir=PipelineConstants.INPUT_IMAGES_DIR,
        target_images_dir=PipelineConstants.TARGET_IMAGES_DIR,
    )

    test_dataset = data_transformer.create_dataset(
        test_df,
        input_images_dir=PipelineConstants.INPUT_IMAGES_DIR,
        target_images_dir=PipelineConstants.TARGET_IMAGES_DIR,
    )

    train_loader = data_transformer.create_data_loader(train_dataset, shuffle=True)
    val_loader = data_transformer.create_data_loader(val_dataset, shuffle=False)
    test_loader = data_transformer.create_data_loader(test_dataset, shuffle=False)

    # Model Training
    model = smp.UnetPlusPlus(
        encoder_name=TrainingConstants.ENCODER_NAME,
        encoder_weights=TrainingConstants.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
    ).to(TrainingConstants.DEVICE)

    model_trainer = ModelTraining(
        model=model,
        loss_fn=MSELoss(),
        optimizer=Adam(model.parameters(), lr=TrainingConstants.LEARNING_RATE),
        experiment_name="experiment 1",
    )

    model_trainer.train(train_loader, val_loader, plot_path="outputs/loss.jpeg")

    model_trainer.save_model(model, "models/experiments/model.pth")
