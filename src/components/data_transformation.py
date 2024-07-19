from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
## pipeline 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd 
import numpy as np
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
import sys,os
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation_obj(self):
        try:
            logging.info('Data Transformation initiated ')

            ## Categorical and Numerical variables 
            categorical_variables=['cut','color','clarity']
            numerical_variables=['carat','depth','table','x','y','z']
            
            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categotries = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]

            logging.info('Data Transformation Pipeline Initialted ')
            #numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())

                ]

            )

            # Categorical Pipeline 
            cat_pipeline = Pipeline(
                steps= [
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categotries])),
                    ('scaler',StandardScaler())
                ]
                
            )
            
            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_variables),
                ('cat_pipeline',cat_pipeline,categorical_variables)
            ])
            
            logging.info('Data Transformation Done ')
            return preprocessor

        except Exception as  e:
            logging.info('Exception'+e+'occured in Data Transformation')
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_data_path,test_data_path ):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Data readed successfully')
            
            preprocessing_obj = self.get_data_transformation_obj()
            
            target_column = 'price'
            drop_column = [target_column,'id']
            ## Getting independent and dependent features
            #Training data 
            input_feature_train_df = train_df.drop(columns=drop_column,axis=1)
            target_feature_train_df=train_df['price']
            # Test data
            
            input_feature_test_df = test_df.drop(columns=drop_column,axis=1)
            target_feature_test_df=test_df['price']
            
            # Applying data transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info('Applying preprocessing object on train and test data')
            train_arr =  np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            


            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            print("^"*40)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info('Exeption '+str(e)+' occured at data transformation')
            raise CustomException(e,sys)