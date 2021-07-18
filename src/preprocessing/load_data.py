import logging
import pandas as pd
from src.preprocessing import prepare_data as data

def load_processed(data_path: str):
    df = pd.read_csv(data_path)
    logging.debug("finished loading, start cleaning")
    df= data.clean_column_names(df)
    logging.debug("cleaned up the column names")
    df= data.remove_duplicate_rows(df)
    logging.debug("dropped duplicates in the data")
    df= data.add_car_age(df)
    logging.debug("created a new column car age")
    df= data.add_car_engine_class(df)
    logging.debug("created a new column engine class for the car")
    df= data.classify_car_model(df)
    logging.debug("classified the car models based on their names")
    df = data.drop_columns(df)
    logging.debug("dropped unnecessary columns")
    return df

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    df = load_processed("../../data/bmw.csv")


