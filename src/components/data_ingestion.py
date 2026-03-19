import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
            self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
         try:
              logging.info("Data ingestion started")
              data = pd.read_csv(os.path.join("notebook", "online_shoppers_intention.csv"))

              os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
              data.to_csv(self.ingestion_config.raw_data_path, index=False)
              logging.info("Raw data saved successfully.")

              train_set, test_set = train_test_split(data, test_size=0.3, random_state=42)  
              logging.info("Data split into train and test sets successfully.")
              
              train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)       
              test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
              
              logging.info("Data ingestion completed successfully.")

              return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path       
         
         except Exception as e:
              logging.info("Error occurred in data ingestion stage")
              raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()