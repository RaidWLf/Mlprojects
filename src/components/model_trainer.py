# Training the model

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression   
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from exception import CustomException
from logger import logging

from utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()    


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Training and testing input data.")
            X_train, y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Ada Boost Classifier": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Cat Boosting Classifier": CatBoostClassifier(verbose=False),
                "Linear Regression": LinearRegression()
            }

            params = { 
                "Random Forest":{
                  
                    'n_estimators':[8,16,32,64,128,256]
                
                },
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                },
                "Ada Boost Classifier":{
                    'n_estimators':[8,16,32,64,128,256],
                    'learning_rate':[0.1,0.01,0.05,0.001],
                },
                "Gradient Boosting":{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256],
                },
                "K-Neighbors Classifier":{
                    'n_neighbors':[5,7,9,11],
                    #'weights':['uniform','distance'],
                    #'algorithm':['auto','ball_tree','kd_tree','brute'],
                },
                "XGBRegressor":{
                    "learning_rate":[0.1,0.01,0.05,0.001],
                    "n_estimators":[8,16,32,64,128,256],
                },
                "Cat Boosting Classifier":{
                    'learning_rate':[0.01,0.05,0.1],
                    'depth':[6,8,10],
                    'iterations':[38,50,100],
                },
                "Linear Regression":{},
                }

            model_report: dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                               models=models,params=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                raise CustomException("No best model found.")
            logging.info(f"Best Model: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)        
        

