import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransfermationConfig:
    preprocesser_obj_file_path = os.path.join('artifacts','preprocesser.pkl')

class DataTransfermation:
    def __init__(self):
        self.data_transfermation_config = DataTransfermationConfig()
    
    def get_data_transfer_object(self):

        """
        this function is resposible for the transformation
        """


        try:
            numercal_columns = ["writing_score","reading_score"]
            categirical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info("Categorical columns standard scaling completed")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categirical_columns}")
            logging.info(f"Numerical columns: {numercal_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numercal_columns),
                    ("cat_pipeline",cat_pipeline,categirical_columns)
                ]
            )
            
            return preprocessor

        except Exception as  e:
            raise CustomException(e,sys)
        
    def initiate_data_transfermation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocesser_obj = self.get_data_transfer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            inpute_fetaure_train_array = preprocesser_obj.fit_transform(input_feature_train_df)
            inpute_fetaure_test_array = preprocesser_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                inpute_fetaure_train_array,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                inpute_fetaure_test_array,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transfermation_config.preprocesser_obj_file_path,
                obj=preprocesser_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transfermation_config.preprocesser_obj_file_path,
            )
                       

        except Exception as e:
            raise CustomException(e,sys)
            
