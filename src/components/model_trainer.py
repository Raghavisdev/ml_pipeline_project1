import os
import sys
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer" , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test =  (
                train_arr[:, :-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
        


        except Exception as e:
            raise CustomException(e,sys)