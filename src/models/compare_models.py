import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from .preprocessing import Dataset, TransformData

from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

class CompareMLBaseline:
    """CompareMLBaseline performs cross validation on the models to spotcheck the ML models that worth more investment
    """    ""
    def __init__(self, data: pd.DataFrame):
        # prepare models
        self.models = []
        self.models.append(('LR', LinearRegression()))
        self.models.append(('LASSO', Lasso()))
        self.models.append(('RIDGE', Ridge()))
        self.models.append(('EN', ElasticNet()))
        self.models.append(('KNN', KNeighborsRegressor()))
        self.models.append(('RF', RandomForestRegressor()))
        self.models.append(('SGD', SGDRegressor()))
        self.models.append(('XGBOOST', XGBRegressor()))
        self.models.append(('SVM', LinearSVR()))
        
        # prepare data
        self.data = Dataset(data)
        self.transform = TransformData()
        self.transform._transform_datasets(self.data.X)
    
    def _get_metric(self, metric="mae"):
        _metrics = {"mae": make_scorer(mean_absolute_error, greater_is_better=False), 
                   "mse": make_scorer(mean_squared_error, greater_is_better=False), 
                   "rmse": make_scorer(mean_squared_error, , greater_is_better=False, squared=False),
                   "r2": make_scorer(r2_score)}
        return _metrics[metric]
        
    def evaluate_models(self, metric="mae", verbose=False):
        self.results = []
        self.names = []
        scoring = self._get_metric(metric)
        
        for name, model in self.models:
            kfold = KFold(n_splits=10)
            cv_results = cross_val_score(model, self.transform.X_train, self.data.y.values, cv=kfold, scoring=scoring)
            self.results.append(cv_results)
            self.names.append(name)
            
            if verbose:
        	    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        	    print(msg)

    def plot_comparison(self, figsize=(15,10)):
        fig = plt.figure(figsize= figsize)
        fig.suptitle('ML models Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(self.results)
        ax.set_xticklabels(self.names)
        plt.show()
