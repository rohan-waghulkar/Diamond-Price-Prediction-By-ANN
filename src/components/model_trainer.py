# # import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas  as pd
import numpy as np
import os,sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from keras.layers import Dense,Dropout
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,build_model,evaluate_model
from dataclasses import dataclass
import keras_tuner as kt


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trianer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            ## Splitting data into Dependent and independent variables 
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info('data Seggregation done into dependent and indenpendent variables')

            ## getting best Model Architecture 
            tunner = kt.RandomSearch(build_model,objective='val_loss',max_trials=10,directory='parameter_tunner',project_name='frist_trial')
            tunner.search(X_train,y_train,epochs=4,validation_split=0.2)

            ## Extracting the best model 
            model=tunner.get_best_models(num_models=1)[0]

            #training the model 
            early_stopping = keras.callbacks.EarlyStopping(monitor='loss',patience=5)
            model.fit(X_train,y_train,epochs=200,initial_epoch=4,callbacks=[early_stopping])
            
            logging.info("model learning Done")
            y_pred = model.predict(X_test)
            logging.info("model prediction for test data done")

            ## model evaluation
            mae,rmse,r2_s,mse=evaluate_model(y_test,y_pred)
            logging.info("\t mean_absolute_error %s\t root_mean_squared_error %s\t mean_Squared_error %s\t r2_score %s", mae, rmse, mse, r2_s)
            logging.info("model evaluation Done")

            save_object(
                file_path=self.model_trianer_config.trained_model_file_path,
                obj=model
            )
        except Exception as e:
            logging.info("Exeption "+str(e)+"Occuerd at model training")
            raise CustomException(e,sys)