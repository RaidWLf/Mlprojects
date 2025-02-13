import os 
import sys 
import dill 
from sklearn.metrics import r2_score



import numpy as np
import pandas as pd

from exception import CustomException
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e,sys)        
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in models:
            model= models[i]

        # for j in params[i]:
        #     param = params[i][j]
            param = params[i]


        
            gs = GridSearchCV(model, param, n_jobs=-1, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

        # model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[i] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)