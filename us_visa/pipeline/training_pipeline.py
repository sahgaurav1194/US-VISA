"""Wires all stages together: ingestion -> validation -> transformation -> training."""

import sys

from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_transformation import DataTransformation
from us_visa.components.data_validation import DataValidation
from us_visa.components.model_trainer import ModelTrainer
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
)
from us_visa.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
)
from us_visa.exception import USvisaException
from us_visa.logger import logging


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise USvisaException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise USvisaException(e, sys) from e

    def start_data_transformation(
        self, data_ingestion_artifact, data_validation_artifact
    ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=self.data_transformation_config,
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise USvisaException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise USvisaException(e, sys) from e

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact, data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            logging.info(f"Pipeline finished. {model_trainer_artifact.metric_artifact}")
        except Exception as e:
            raise USvisaException(e, sys) from e


if __name__ == "__main__":
    TrainPipeline().run_pipeline()
