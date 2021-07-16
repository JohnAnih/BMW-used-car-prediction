from typing import List
from datetime import datetime as dt
import numpy as np
import pandas as pd

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower()
    return df

def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def add_car_age(df: pd.DataFrame) -> pd.DataFrame:
    current_year = dt.now().year
    df["car_age"] = current_year - df["year"]
    return df

def add_car_engine_class(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 1, 2, 3, np.inf]
    names = ["under 1.0-litre Engine", "1.0 to 2.0-litres Engine", "2.1 to 3.0-litres Engine", "Above 3.0-litres Engine"]
    df['engine_class'] = pd.cut(df['enginesize'], bins, labels=names)
    return df

def classify_car_model(df: pd.DataFrame) -> pd.DataFrame:
    series_model = {model: "series_model" for model in [f" {i} Series" for i in range(1,9)]}
    x_model = {model: "x_model" for model in [f" X{i}" for i in range(1,9)]}
    m_model = {model: "m_model" for model in [f" M{i}" for i in range(1,9)]}
    i_model = {model: "i_model" for model in [f" i{i}" for i in range(1,9)]}
    z_model = {model: "z_model" for model in [f" Z{i}" for i in range(1,9)]}
    
    model_type = {**series_model, **x_model, **m_model, **i_model, **z_model}
    df["model_type"] = df["model"].replace(model_type)
    return df

def drop_columns(df: pd.DataFrame, columns: List[str] = None):
    if columns is None:
        columns = ["model", "year", "enginesize"]

    return df.drop(labels=columns, axis=1)
    