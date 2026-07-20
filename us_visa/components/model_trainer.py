"""
Stage 4 — Model Trainer

Trains and tunes the candidate models from the CV stack
(Logistic Regression, Random Forest, Decision Tree) with GridSearchCV,
keeps the best one, evaluates it on the test split, and only accepts it
if accuracy clears the expected threshold. The final artifact is a
USvisaModel = fitted preprocessor + best model in one object.
"""

import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from us_visa.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.estimator import USvisaModel
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_numpy_array_data, load_object, save_object

MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {"C": [0.1, 1, 10]},
    ),
    "decision_tree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [5, 10, None], "min_samples_split": [2, 5]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [10, None]},
    ),
}


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(),
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_best_model(self, train: np.ndarray) -> Tuple[str, object, float]:
        """Runs GridSearchCV over every candidate and returns the winner."""
        try:
            x_train, y_train = train[:, :-1], train[:, -1]

            best_name, best_model, best_score = None, None, -1.0
            for name, (model, param_grid) in MODELS.items():
                logging.info(f"Tuning {name} ...")
                gs = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
                gs.fit(x_train, y_train)
                logging.info(f"{name}: best cv f1 = {gs.best_score_:.4f} with {gs.best_params_}")
                if gs.best_score_ > best_score:
                    best_name, best_model, best_score = name, gs.best_estimator_, gs.best_score_

            logging.info(f"Best model: {best_name} (cv f1 = {best_score:.4f})")
            return best_name, best_model, best_score
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info(">>> Model Trainer started <<<")
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            best_name, best_model, _ = self.get_best_model(train_arr)

            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            y_pred = best_model.predict(x_test)

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1_score(y_test, y_pred),
                precision_score=precision_score(y_test, y_pred),
                recall_score=recall_score(y_test, y_pred),
                accuracy_score=accuracy_score(y_test, y_pred),
            )
            logging.info(f"Test metrics for {best_name}: {metric_artifact}")

            if metric_artifact.accuracy_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Best model accuracy {metric_artifact.accuracy_score:.4f} is below the "
                    f"expected threshold {self.model_trainer_config.expected_accuracy}"
                )

            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            usvisa_model = USvisaModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model,
            )
            save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f">>> Model Trainer completed: {model_trainer_artifact} <<<")
            return model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
