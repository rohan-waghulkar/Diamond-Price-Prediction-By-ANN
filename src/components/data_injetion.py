import os 
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
## initialize the data ingetion configuration

@dataclass
class DataIngetionConfig:
    train_data_path:str=os.path.join('artifacts','train_data.csv')
    test_data_path:str=os.path.join('artifacts','test_data.csv')
    raw_data_path:str=os.path.join('artifacts','raw_data.csv')

## creating the data ingetion class 

class DataIngetion:
    def __init__(self):
        self.ingetion_config=DataIngetionConfig()
    def initiate_data_ingetion(self):
        logging.info("Data Ingetion starts")
        try:
            df=pd.read_csv(os.path.join('notebook/data','gemstone.csv'))
            logging.info('dataset loaded')
            os.makedirs(os.path.dirname(self.ingetion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingetion_config.raw_data_path,index=False)

            train_set,test_set=train_test_split(df,test_size=0.3,random_state=32)

            train_set.to_csv(self.ingetion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingetion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingetion completed')

            return (
                self.ingetion_config.train_data_path,
                self.ingetion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception '+e+' occured at data ingetion')
            raise CustomException(e,sys)