import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from exception import CustomException
from logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_tranformer_object(self):

    # This function is responsble for the data transformation.

        try:
            numerical_columns = ['writing score','reading score']
            categorical_columns = ['gender','race_ethinicity','parental level of education','lunch','test preparation course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('std_scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder()),
                    ('scaling', StandardScaler())
                ]
            )
            
            logging.info("numerical and categorical pipeline created successfully.")

            full_pipeline = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Data Transformation object created successfully")

            return full_pipeline


        except Exception as e:
            raise CustomException(e,sys)
        

    
