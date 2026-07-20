"""
USvisaModel bundles the fitted preprocessor and the trained model into a
single object, so prediction code never has to worry about applying the
right transformations — it just calls .predict() on raw input.
"""

import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from us_visa.exception import USvisaException
from us_visa.logger import logging


class TargetValueMapping:
    def __init__(self):
        self.Certified: int = 0
        self.Denied: int = 1

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class USvisaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame):
        try:
            transformed_feature = self.preprocessing_object.transform(dataframe)
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return self.__repr__()
