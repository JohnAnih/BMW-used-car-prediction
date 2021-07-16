from abc import ABC, abstractmethod
from typing import Callable, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

from .settings import SEED
from .preprocessing import Dataset, TransformData


class TuneMLAlgorithms(ABC):
    """TuneMLAlgorithms is designed to create an interface for several types of HyperParameter Tuning Methods

    Args:
        data (pd.DataFrame): The pandas DataFrame for the predictive modelling
        estimator (sklearn estimator): scikit-learn model estimator to optimize
        params (Dict): parameters to optimize
        cv (int): Determines the cross-validation splitting strategy
        metric (str): Evaluation metric to evaluate model performance

    Attributes:
        model (estimator): The trained model
        train_predictions (pd.DataFrame): predictions vs groundtruth on the training set
        test_predictions (pd.DataFrame): predictions vs groundtruth on the test set

    """
    def __init__(self, data: pd.DataFrame, estimator: Callable, params: Dict, cv: int, metric: str="mse"):
        self.estimator = estimator()
        self.params = params
        self.scoring = self._get_metric(metric)
        self.cv = cv
        
        # set seed for reproducibility and use on 4 cores of CPU
        self.random_state = SEED
        self.return_train_score = True
        self.n_jobs = 4
        
        # load, split and prepare the data
        self.data = Dataset(data)
        self.transform = TransformData()
        self.data.split_datasets()
        self._prepare_inputs()
    
    def _prepare_inputs(self):
        self.transform._transform_datasets(self.data.X_train, self.data.X_test)
    
    def _get_metric(self, metric: str):
        # custom scoring metric to evaluate model performance
        _metrics = {"mae": make_scorer(mean_absolute_error), 
                   "mse": make_scorer(mean_squared_error), 
                   "rmse": make_scorer(mean_squared_error, squared=False),
                   "r2": make_scorer(r2_score)}
        return _metrics[metric]
    
    @abstractmethod
    def _create_model(self):
        # This creates the model for the hyperparameter type
        pass
    
    def validate(self):
        self._train_predictions = self.model.predict(self.transform.X_train)
        self._train_actual = self.data.y_train.values
        self._test_predictions = self.model.predict(self.transform.X_test)
        self._test_actual = self.data.y_test.values
        
        self.train_predictions = pd.DataFrame()
        self.test_predictions = pd.DataFrame()
        
        self.train_predictions["predictions"] = self._train_predictions
        self.train_predictions["actual"] = self._train_actual
        
        self.test_predictions["predictions"] = self._test_predictions
        self.test_predictions["actual"] = self._test_actual
        
    def predict(self, sample: pd.DataFrame)-> np.array:
        """The predict method allows for predictions on new samples
        
        Args:
            sample (pd.DataFrame): Contains one observation to make prediction
        """        ""
        sample = self.transform._transform_data(sample)
        return self.model.predict(sample)
    
    def _evaluate_on(self, metric, y_true, y_pred):
        _metrics = {"mae": mean_absolute_error(y_true, y_pred), 
                    "mse": mean_squared_error(y_true, y_pred), 
                    "rmse": mean_squared_error(y_true, y_pred, squared= False) , 
                    "r2": r2_score(y_true, y_pred)
            }
        return _metrics[metric]
    
    def evaluate_model(self, metric):
        try:
            train_score = self._evaluate_on(metric=metric, y_true=self._train_actual, y_pred=self._train_predictions)
            test_score = self._evaluate_on(metric=metric, y_true=self._test_actual, y_pred=self._test_predictions)
            print(f"The Train {metric}: {train_score}")
            print(f"The Test {metric}: {test_score}")
        except AttributeError:
            print("No ground truth to compare with")
            
    def search_best(self):
        self.model = self._create_model()
        self.model.fit(self.transform.X_train, self.data.y_train)
        return self.model.best_score_, self.model.best_params_


class GridSearchTuner(TuneMLAlgorithms):
    def __init__(self, data: pd.DataFrame, estimator: Callable, params: Dict, cv: int, metric: str="mse", **kwargs):
        super(GridSearchTuner, self).__init__(data, estimator, params, cv, metric)
        self.kwargs = kwargs
        
    def _create_model(self):
        model = GridSearchCV(estimator = self.estimator, param_grid = self.params, scoring = self.scoring, 
                             n_jobs = self.n_jobs, cv = self.cv, return_train_score = self.return_train_score, **self.kwargs)
        return model
    

class RandomSearchTuner(TuneMLAlgorithms):
    def __init__(self, data: pd.DataFrame, estimator: Callable, params: Dict, cv: int, n_iter: int, metric: str="mse", **kwargs):
        super(RandomSearchTuner, self).__init__(data, estimator, params, cv, metric)
        self.n_iter = n_iter
        self.kwargs = kwargs

    def _create_model(self):
        model  = RandomizedSearchCV(estimator = self.estimator, param_distributions = self.params, scoring = self.scoring, 
                                    n_iter = self.n_iter, n_jobs = self.n_jobs, cv = self.cv, random_state = self.random_state,
                                    return_train_score = self.return_train_score, **self.kwargs)
        return model


class BayesianSearchTuner(TuneMLAlgorithms):
    def __init__(self, data: pd.DataFrame, estimator: Callable, params: Dict, cv: int, n_iter: int, metric: str="mse", **kwargs):
        super(BayesianSearchTuner, self).__init__(data, estimator, params, cv, metric)
        self.n_iter = n_iter
        self.kwargs = kwargs

    def _create_model(self):
        model = BayesSearchCV(estimator = self.estimator, search_spaces = self.params, scoring = self.scoring, 
                              n_iter = self.n_iter, n_jobs = self.n_jobs, cv = self.cv, random_state = self.random_state, 
                              return_train_score = self.return_train_score, **self.kwargs)
        return model