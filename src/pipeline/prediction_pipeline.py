import os 
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd 


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,freatures):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            logging.info("pickled object loaded ")
            
            data_scaled = preprocessor.transform(freatures)
            logging.info("prediction data preprocessing completed")

            pred = model.predict(data_scaled)
            logging.info("Prediction Successfull")
            return pred
        except Exception as e:
            logging.info('Error'+str(e)+"occured while prediction")
            raise CustomException(e,sys)


class Customdata:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "carat": [self.carat], 
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z":[ self.z]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('dataframe created')

            return df
        
        
        except Exception as e :
            logging.info('Error'+str(e)+"occured while creating a dataframe from data")
            raise CustomException(e,sys)