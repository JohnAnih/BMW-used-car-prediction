import pickle as pkl
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor

from .base_estimator import BaseEstimator
from .settings import SEED


class MLBase(BaseEstimator):
    def __init__(self, data: pd.DataFrame):
        super(MLBase, self).__init__(data)
        self.random_state = SEED
    
    def prepare_inputs(self):
        self.transform._transform_datasets(self.data.X_train, self.data.X_test)
    
    def train(self):
        self.prepare_inputs()
        self.model.fit(self.transform.X_train, self.data.y_train)
    
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

    def predict(self, sample):
        sample = self.transform._transform_data(sample)
        return self.model.predict(sample)
    
    def _get_metric(self, metric, y_true, y_pred):
        _metrics = {"mae": mean_absolute_error(y_true, y_pred), 
                    "mse": mean_squared_error(y_true, y_pred), 
                    "rmse": mean_squared_error(y_true, y_pred, squared= False) , 
                    "r2": r2_score(y_true, y_pred)
            }
        return _metrics[metric]
        
    def evaluate_model(self, metric="mae"):
        try:
            train_score = self._get_metric(metric=metric, y_true=self._train_actual, y_pred=self._train_predictions)
            test_score = self._get_metric(metric=metric, y_true=self._test_actual, y_pred=self._test_predictions)
            print(f"The Train {metric}: {train_score}")
            print(f"The Test {metric}: {test_score}")
        except AttributeError:
            print("No ground truth to compare with")
    
    def get_feature_importance(self, plot=True, figsize=(15,10)):
        try:
            feature_ranking= pd.Series(self.model.feature_importances_, index=self.transform.train_inputs.columns)
        except:
            raise NotImplemented("The model feature importance has not been implemented yet")
        if plot:
            feature_ranking.sort_values().plot(kind="barh", figsize=figsize)
    
    def save_model(self, filepath):
        pkl.dump(self, open(filepath, 'wb'))
        print(f"Successfully saved model -> {filepath}")



class MedianPredictor(MLBase):
    def __init__(self, data: pd.DataFrame):
        super(MedianPredictor, self).__init__(data)
        self.model = DummyRegressor(strategy="median")


class MeanPredictor(MLBase):
    def __init__(self, data: pd.DataFrame):
        super(MeanPredictor, self).__init__(data)
        self.model = DummyRegressor(strategy="mean")


class MostFrequentPredictor(MLBase):
    def __init__(self, data: pd.DataFrame):
        super(MostFrequentPredictor, self).__init__(data)
        mode = self.data.y_train.mode().values[0]
        self.model = DummyRegressor(constant=mode)


class XGBoostRegressor(MLBase):
    def __init__(self, data: pd.DataFrame, **kwargs):
        super(XGBoostRegressor, self).__init__(data)
        self.model = XGBRegressor(**kwargs)
    
    def get_feature_importance(self, plot=True, figsize=(15,10)):
        pass


class KNN(MLBase):
    """Implementation of KNN ML model"""

    def __init__(self, data: pd.DataFrame, **kwargs):
        super(KNN, self).__init__(data)
        self.model = KNeighborsRegressor(**kwargs)


class RandomForest(MLBase):
    """Implementation of Random Forest ML model"""

    def __init__(self, data: pd.DataFrame, **kwargs):
        super(RandomForest, self).__init__(data)
        self.model = RandomForestRegressor(**kwargs)

    def get_feature_importance(self, plot=True, figsize=(15,10)):
        pass


class StochasticGradientRegressor(MLBase):
    """Implementation of SGD Model"""

    def __init__(self, data: pd.DataFrame, **kwargs):
        super(StochasticGradientRegressor, self).__init__(data)
        self.model = SGDRegressor(**kwargs)

    def get_feature_importance(self, plot=True, figsize=(15,10)):
        pass


class EnsembleModels(MLBase):
    """Ensemble learning models allows the combination of best performing models"""

    def __init__(self, data: pd.DataFrame, **kwargs):
        super(EnsembleModels, self).__init__(data)
        
        self.xgboost = XGBRegressor(colsample_bytree= 0.6245078326379802, max_dept=15, 
                               learning_rate=0.07702990702840093, n_estimators=300, 
                               subsample=0.8416508000885121, verbosity=0, random_state= self.random_state)
        
        self.rf = RandomForestRegressor(max_depth= 20, min_samples_leaf= 1, min_samples_split=5, 
                                        n_estimators=400, random_state= self.random_state)
        
        self.estimators = [('xgboost', self.xgboost), ('rf', self.rf)]
        
        self.model = VotingRegressor(estimators=self.estimators, **kwargs)
    
    def _get_models_rankings(self):
        self.xgboost.fit(self.transform.X_train, self.data.y_train)
        self.rf.fit(self.transform.X_train, self.data.y_train)
        
    def get_feature_importance(self, plot=True, figsize=(15,10)):
        self._get_models_rankings()
        feature_ranking = pd.DataFrame()
        
        feature_ranking["features"] = self.transform.train_inputs.columns
        feature_ranking["RandomForest"] = self.rf.feature_importances_
        feature_ranking["XGBoost"] = self.xgboost.feature_importances_
        feature_ranking["AvgRanking"] = feature_ranking.sum(axis=1)
        
        if plot:
            feature_ranking.sort_values("AvgRanking").plot(x="features", kind="barh", figsize=figsize)
   

