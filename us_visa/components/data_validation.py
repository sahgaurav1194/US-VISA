"""
Stage 2 — Data Validation

Checks that the ingested train/test files match the expected schema
(config/schema.yaml): correct number of columns and all required
numerical / categorical columns present. Writes a YAML report either way.
"""

import json
import os
import sys

import pandas as pd

from us_visa.constants import SCHEMA_FILE_PATH
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig = DataValidationConfig(),
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column count present: [{status}]")
            return status
        except Exception as e:
            raise USvisaException(e, sys)

    def is_column_exist(self, df: pd.DataFrame) -> bool:
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = [
                col for col in self._schema_config["numerical_columns"]
                if col not in dataframe_columns
            ]
            missing_categorical_columns = [
                col for col in self._schema_config["categorical_columns"]
                if col not in dataframe_columns
            ]

            if missing_numerical_columns:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")
            if missing_categorical_columns:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return not (missing_numerical_columns or missing_categorical_columns)
        except Exception as e:
            raise USvisaException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        logging.info(">>> Data Validation started <<<")
        try:
            validation_error_msg = ""
            train_df = DataValidation.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = DataValidation.read_data(self.data_ingestion_artifact.test_file_path)

            for name, df in [("training", train_df), ("test", test_df)]:
                if not self.validate_number_of_columns(dataframe=df):
                    validation_error_msg += f"Columns are missing in {name} dataframe. "
                if not self.is_column_exist(df=df):
                    validation_error_msg += f"Required columns are missing in {name} dataframe. "

            validation_status = len(validation_error_msg) == 0
            message = validation_error_msg if validation_error_msg else "Validation successful"

            write_yaml_file(
                file_path=self.data_validation_config.validation_report_file_path,
                content={"validation_status": validation_status, "message": message},
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=message,
                validation_report_file_path=self.data_validation_config.validation_report_file_path,
            )

            if not validation_status:
                raise Exception(f"Data validation failed: {message}")

            logging.info(f">>> Data Validation completed: {data_validation_artifact} <<<")
            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
