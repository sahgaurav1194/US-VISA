"""Takes raw form input, builds a DataFrame, and predicts with the saved USvisaModel."""

import sys

import pandas as pd

from us_visa.constants import CURRENT_YEAR
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.estimator import TargetValueMapping
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_object


class USvisaData:
    def __init__(
        self,
        continent,
        education_of_employee,
        has_job_experience,
        requires_job_training,
        no_of_employees,
        region_of_employment,
        prevailing_wage,
        unit_of_wage,
        full_time_position,
        company_age,
    ):
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_usvisa_input_data_frame(self) -> pd.DataFrame:
        try:
            return pd.DataFrame(
                {
                    "continent": [self.continent],
                    "education_of_employee": [self.education_of_employee],
                    "has_job_experience": [self.has_job_experience],
                    "requires_job_training": [self.requires_job_training],
                    "no_of_employees": [self.no_of_employees],
                    "region_of_employment": [self.region_of_employment],
                    "prevailing_wage": [self.prevailing_wage],
                    "unit_of_wage": [self.unit_of_wage],
                    "full_time_position": [self.full_time_position],
                    "company_age": [self.company_age],
                }
            )
        except Exception as e:
            raise USvisaException(e, sys) from e


class USvisaClassifier:
    def __init__(self):
        try:
            self.model_path = ModelTrainerConfig().trained_model_file_path
        except Exception as e:
            raise USvisaException(e, sys)

    def predict(self, dataframe: pd.DataFrame) -> str:
        try:
            logging.info("Running prediction")
            model = load_object(self.model_path)
            result = model.predict(dataframe)[0]
            return TargetValueMapping().reverse_mapping()[int(result)]
        except Exception as e:
            raise USvisaException(e, sys)
