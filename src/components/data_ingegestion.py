import os
import sys 
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionconfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()
    
    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/data.csv')
            logging.info("Exported and read the dataset as dataframe.")
            
            pass
        except:
            pass    


    