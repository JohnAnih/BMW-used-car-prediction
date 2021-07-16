from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Union

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import Dataset, TransformData


class BaseEstimator(ABC):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = Dataset(data)
        self.transform = TransformData()
        self.data.split_datasets()
    
    @abstractmethod
    def prepare_inputs(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def validate(self):
        pass
    
    @abstractmethod
    def get_feature_importance(self):
        pass

    @abstractmethod
    def predict(self, sample: Union[pd.DataFrame, np.array]):
        pass
            
        