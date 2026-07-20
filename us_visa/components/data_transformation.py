"""
Stage 3 — Data Transformation

Feature engineering + preprocessing:
  * derive company_age from yr_of_estab
  * one-hot encode nominal columns, ordinal encode education
  * power transform skewed columns, scale numerical columns
  * handle class imbalance with SMOTEENN
The fitted preprocessor is saved so the exact same transformations are
applied at prediction time.
"""

import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler

from us_visa.constants import CURRENT_YEAR, SCHEMA_FILE_PATH, TARGET_COLUMN
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.estimator import TargetValueMapping
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import (
    read_yaml_file,
    save_numpy_array_data,
    save_object,
)


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig = DataTransformationConfig(),
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown="ignore")
            ordinal_encoder = OrdinalEncoder()

            oh_columns = self._schema_config["oh_columns"]
            or_columns = self._schema_config["or_columns"]
            transform_columns = self._schema_config["transform_columns"]
            num_features = self._schema_config["num_features"]

            transform_pipe = Pipeline(steps=[("transformer", PowerTransformer(method="yeo-johnson"))])

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features),
                ]
            )
            logging.info("Created preprocessor object from ColumnTransformer")
            return preprocessor
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info(">>> Data Transformation started <<<")
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            preprocessor = self.get_data_transformer_object()

            train_df = DataTransformation.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)

            # ---- feature engineering: company_age, drop id/raw year ----
            drop_cols = self._schema_config["drop_columns"]

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_train_df["company_age"] = CURRENT_YEAR - input_feature_train_df["yr_of_estab"]
            input_feature_train_df = input_feature_train_df.drop(columns=drop_cols)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]
            input_feature_test_df["company_age"] = CURRENT_YEAR - input_feature_test_df["yr_of_estab"]
            input_feature_test_df = input_feature_test_df.drop(columns=drop_cols)

            target_feature_train_df = target_feature_train_df.replace(
                TargetValueMapping()._asdict()
            ).astype(int)
            target_feature_test_df = target_feature_test_df.replace(
                TargetValueMapping()._asdict()
            ).astype(int)

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # ---- handle class imbalance ----
            logging.info("Applying SMOTEENN on training dataset")
            smt = SMOTEENN(sampling_strategy="minority", random_state=42)
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(">>> Data Transformation completed <<<")
            return data_transformation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
