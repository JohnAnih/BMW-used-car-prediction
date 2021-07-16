from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from .onehotcoder import OneHotEncoder
from .settings import SEED


@dataclass
class Dataset:
    data: pd.DataFrame
    
    def __post_init__(self):
        self.X =  self.data.drop("price", axis=1)
        self.y = self.data["price"]
    
    def split_datasets(self):
        np.random.seed(SEED)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=SEED)
    

class TransformData:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ordinal_encoder = OrdinalEncoder()
        self.onehot_encoder = OneHotEncoder()
        self.oe_columns = ["transmission", "fueltype"]
        self.ohe_columns = ["engine_class", "model_type"]
    
        
    def _transform_datasets(self, X_train, X_test=None):
        X_train[self.oe_columns] = self.ordinal_encoder.fit_transform(X_train[self.oe_columns])
        encoded_data = self.onehot_encoder.fit_transform(X_train[self.ohe_columns])
        X_train = X_train.drop(self.ohe_columns, axis=1)
        
        # store train data
        self.X_train = pd.concat([X_train, encoded_data], axis=1)
        
        # get the inputs in readable inputs before scaling
        self.train_inputs = pd.DataFrame(self.X_train, index = self.X_train.index, columns = self.X_train.columns)
    
        # scale the data
        self.X_train = self.scaler.fit_transform(self.X_train)
        
        # transform test data
        if X_test is not None:
            self.X_test = self._transform_data(X_test)
    
    def _transform_data(self, X_test):
        X_test[self.oe_columns] = self.ordinal_encoder.transform(X_test[self.oe_columns])
        encoded_data = self.onehot_encoder.transform(X_test[self.ohe_columns])
        X_test = X_test.drop(self.ohe_columns, axis=1)
        X_test = pd.concat([X_test, encoded_data], axis=1)
        # get the data in readable inputs before scaling
        self.test_inputs = pd.DataFrame(X_test, index=X_test.index, columns= X_test.columns)

        self.X_test = self.scaler.transform(X_test)
        
        return self.X_test