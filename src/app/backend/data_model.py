import locale
from datetime import datetime as dt

import pickle as pkl
import pandas as pd

locale.setlocale(locale.LC_ALL, '')


class DataModel:
    def __init__(self, mileage: int, transmission: str, fuel_type: str, car_year: int, 
                 mpg: int, tax: int, engine_class: str, model_type: str, model_filepath: str)-> None:
        self.mileage = mileage
        self.transmission = transmission
        self.fuel_type = fuel_type
        self.car_year = car_year
        self.mpg = mpg
        self.tax = tax
        self.engine_class = engine_class
        self.model_type = model_type
        
        # prepare data
        self._preprocess_data()
        
        # load model
        self._load_model(model_filepath)
        
    def _load_model(self, filepath: str)-> None:
        self.model = pkl.load(open(filepath, 'rb'))
    
    def _create_datapoint(self)-> pd.DataFrame:
        return pd.DataFrame({"transmission": [self.transmission], 
                             "mileage": [self.mileage], 
                             "fueltype": [self.fuel_type], 
                             "tax": [self.tax], 
                             "mpg": [self.mpg], 
                             "car_age": [self.car_age], 
                             "engine_class": [self.engine_class], 
                             "model_type": [self.model_type]})
    
    def _preprocess_data(self)-> None:
        # get the car model in the keys it was trained on
        car_models = {
            "X series": "x_model", "M series": "m_model", 
            "I series": "i_model", "Z series": "z_model"
        }
        self.model_type = car_models[self.model_type]
        
        # get the car age instead of the car year
        self.car_age = dt.now().year - self.car_year

    def predict_car_price(self)-> str:
        # get the input data ready
        sample = self._create_datapoint()
        prediction = self.model.predict(sample)[0].round(2)
        return f'{prediction:n}'