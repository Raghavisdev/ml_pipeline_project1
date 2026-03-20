import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer 
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts/data_transformation" , "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_obj = DataTransformationConfig()

    def get_data_transformation_obj(self):
        # Encoded , Scaled
        try:
            logging.info(" Data Tranformation Started ")

            num_cols = [
            'Administrative', 'Administrative_Duration',
            'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration',
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
            ]

            cat_cols = ['Month', 'VisitorType']

            coded_cat_cols = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']

            bool_cols = ['Weekend', 'Revenue']

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num",StandardScaler(), num_cols),
                    ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols),
                    ("bool","passthrough", bool_cols)
                ]

            )
            logging.info("Encoded and Scaled Successfully")

            return preprocessor

        except Exception as e:

            raise CustomException
        

    def inititiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            
        except Exception as e:

            raise CustomException




