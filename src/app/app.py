import sys
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from frontend.home_page import create_navbar, create_home_page
from frontend.prediction import create_car_info_page
from frontend.utils import get_results, show_errror_msg
from frontend.data_exploration import DataExploration

from backend.data_model import DataModel
from backend.plotting import plot_data

sys.path.append("../../src")
from models.estimators import EnsembleModels
from preprocessing.load_data import load_processed

MODEL_PATH = "../models/model_store/model.pkl"
DATA = load_processed("../../data/bmw.csv")

app = dash.Dash(__name__, title="BMW Used Car Prediction", 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                #update_title="BMW Used Car Prediction"
                )

app.config.suppress_callback_exceptions = True


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    create_navbar(),

    # content will be rendered in this element
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/":
        return create_home_page()
    
    elif pathname == "/data-exploration":
        return DataExploration().layout
    
    elif pathname == "/predict-a-used-BMW-car":
        return create_car_info_page()
    
    else:
        return "404 Not Found"


@app.callback(
    Output('model-results', 'children'),
    Output("user_response", "is_open"),
    Input("submit-btn", "n_clicks"),
    Input("close", "n_clicks"),
    State("mileage", "value"),
    State("transmission", "value"),
    State("fuel_type", "value"),
    State("car_year", "value"),
    State("mpg", "value"),
    State("tax", "value"),
    State("engine_class", "value"),
    State("model_type", "value"),
    State("user_response", "is_open"), 
    prevent_initial_call=True
)
def make_prediction(click_submit, close_btn, mileage, transmission, fuel_type, 
                    car_year, mpg, tax, engine_class, model_type, is_open):
    global MODEL_PATH
    
    try:
        data_model = DataModel(mileage=mileage, transmission=transmission, 
                               fuel_type=fuel_type, car_year=car_year, 
                               mpg=mpg, tax=tax, engine_class=engine_class, 
                               model_type=model_type, model_filepath=MODEL_PATH)
        prediction = data_model.predict_car_price()

        results = get_results(mileage, transmission, fuel_type, car_year, mpg, tax, 
                              engine_class, model_type, prediction)
    except:
        results = show_errror_msg()
        
    if click_submit or close_btn:
        return results, not is_open
    
    return results, is_open
    
@app.callback(
    Output('plot_data', 'figure'),
    Input("features", "value"),
    Input("hue", "value"),
    Input("plot_type", "value")
)
def plot_historical_data(feature, split_by, plot_type):
    global DATA
    return plot_data(DATA, feature, split_by, plot_type)

if __name__ == '__main__':
    app.run_server(debug=False)