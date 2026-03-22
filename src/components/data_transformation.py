import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts", "data_transformation", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Creating preprocessing pipelines")


            num_cols = [
                'Administrative', 'Administrative_Duration',
                'Informational', 'Informational_Duration',
                'ProductRelated', 'ProductRelated_Duration',
                'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
            ]

    
            cat_cols = ['Month', 'VisitorType']
            coded_cat_cols = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']


            bool_cols = ['Weekend']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("log_transform", FunctionTransformer(np.log1p)),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )

            # Column Transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, num_cols),
                    ("cat", cat_pipeline, cat_cols + coded_cat_cols),
                    ("bool", "passthrough", bool_cols)
                ]
            )

            logging.info("Preprocessing Pipeline Created Successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            preprocessor_obj = self.get_data_transformation_obj()

            target_column = "Revenue"

            # Split input and target
            input_feature_train_data = train_data.drop(columns=[target_column])
            target_feature_train_data = train_data[target_column]

            input_feature_test_data = test_data.drop(columns=[target_column])
            target_feature_test_data = test_data[target_column]

            logging.info("Applying preprocessing on training data")
            input_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)

            logging.info("Applying preprocessing on testing data")
            input_test_arr = preprocessor_obj.transform(input_feature_test_data)

            train_arr = np.c_[input_train_arr, target_feature_train_data.to_numpy()]
            test_arr = np.c_[input_test_arr, target_feature_test_data.to_numpy()]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Preprocessor saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)