"""
Stage 1 — Data Ingestion

Pulls the raw visa dataset from MongoDB Atlas, stores a copy in the
feature store, then splits it into train / test sets.

For local runs without MongoDB access, it falls back to the sample CSV
kept under notebooks/ so the whole pipeline stays runnable end to end.
"""

import os
import sys

from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split

from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.exception import USvisaException
from us_visa.logger import logging

LOCAL_SAMPLE_PATH = os.path.join("notebooks", "EasyVisa.csv")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise USvisaException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Exporting data from MongoDB")
            if os.getenv("MONGODB_URL"):
                from us_visa.data_access.usvisa_data import USvisaData

                usvisa_data = USvisaData()
                dataframe = usvisa_data.export_collection_as_dataframe(
                    collection_name=self.data_ingestion_config.collection_name
                )
            else:
                logging.warning("MONGODB_URL not set — falling back to local sample CSV")
                dataframe = read_csv(LOCAL_SAMPLE_PATH)

            logging.info(f"Shape of dataframe: {dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise USvisaException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
            )
            logging.info("Performed train test split on the dataframe")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Exported train and test file paths.")
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info(">>> Data Ingestion started <<<")
        try:
            dataframe = self.export_data_into_feature_store()
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f">>> Data Ingestion completed: {data_ingestion_artifact} <<<")
            return data_ingestion_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
