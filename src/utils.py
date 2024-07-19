import os
import sys
import pickle
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import keras_tuner as kt
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from keras.layers import Dense,Dropout

def load_object(path):
    try:
        with open(path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Error'+str(e)+"occured while loding pickle object")
        raise CustomException(e,sys)



def evaluate_model(test,predicted):
    mae=mean_absolute_error(test,predicted)
    mse=mean_squared_error(test,predicted)
    rmse=np.sqrt(mean_squared_error(test,predicted))
    r2_S=r2_score(test,predicted)
    return mae,rmse,r2_S,mse


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb")as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    


## Hyperparameter tunning 
# from itertools import count

def build_model(hp):
  try:
    model=Sequential()
    counter=0
    for i in range(hp.Int('num_layers',min_value=1,max_value=10)):
        if counter==0:
            model.add(
                Dense(
                    units=hp.Int('units'+str(i),min_value=32,max_value=512,step=32),
                    activation='relu',
                    input_dim=9
                )
            )
        else:
            model.add(
                Dense(
                    units=hp.Int('units'+str(i),min_value=32,max_value=512,step=32),
                    activation='relu'
                )
            )
        counter+=1
    model.add(
        Dense(1)
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning rate',values=[1e-2,1e-3,1e-4])
        ),

        loss='mean_absolute_error',
        # metrics=[r2_score]
    )
    return model
  except Exception as e:
      logging.info("Exeption "+str(e)+"occured at build_model method")
      raise CustomException(e,sys)