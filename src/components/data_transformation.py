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
from utils import save_object

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
            categorical_columns = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('std_scaler', StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder()),
                    ('scaling', StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("numerical and categorical pipeline created successfully.")

            preprocessor = ColumnTransformer(
                # full pipeline
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Data Transformation object created successfully")

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformer(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train and test data successful.")

            logging.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_data_tranformer_object()
            target_column_name = 'math score'
            numerical_columns = ['writing score','reading score']
            
            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("fitting the preprocessor object on train data and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[ 
                input_feature_train_arr, 
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[ 
                input_feature_test_arr, 
                np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing objects.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        


        except Exception as e:
            raise CustomException(e,sys)

