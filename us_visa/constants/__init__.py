"""
All project level constants live here so nothing is hard-coded
inside the components.
"""

import os
from datetime import date

# ---------------- MongoDB ----------------
DATABASE_NAME = "US_VISA"
COLLECTION_NAME = "visa_data"
MONGODB_URL_KEY = "MONGODB_URL"

# ---------------- Pipeline ----------------
PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"

FILE_NAME: str = "usvisa.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

MODEL_FILE_NAME = "model.pkl"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

TARGET_COLUMN = "case_status"
CURRENT_YEAR = date.today().year

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# ---------------- Data Ingestion ----------------
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# ---------------- Data Validation ----------------
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

# ---------------- Data Transformation ----------------
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# ---------------- Model Trainer ----------------
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6

# ---------------- App ----------------
APP_HOST = "0.0.0.0"
APP_PORT = 8080
