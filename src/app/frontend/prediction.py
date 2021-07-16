import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq

__all__ = ["create_car_info_page"]

TRANSMISSION = ['Automatic', 'Manual', 'Semi-Auto']
FUEL_TYPE = ['Diesel', 'Petrol', 'Other', 'Hybrid', 'Electric']
ENGINE_CLASS = ["under 1.0-litre Engine", "1.0 to 2.0-litres Engine", "2.1 to 3.0-litres Engine", "Above 3.0-litres Engine"]
CAR_MODEL_TYPE = ["X series", "M series", "I series", "Z series"] 

def _user_mileage_info():
    return dbc.FormGroup(
             [
                 dbc.Label("Mileage", html_for="mileage", width=2, color="white"),
                 dbc.Col(
                     dbc.Input(
                         type="number", id="mileage", placeholder="Enter your car mileage"
                     ),
                     width=7,
                 ),
             ],
             row=True,
             className= "pad-bottom"
         )

def _user_car_transmission():
    return dbc.FormGroup(
             [
                 dbc.Label("Car Transmission", html_for="transmission", width=2, color="white"),
                 dbc.Col(
                     dbc.RadioItems(
                         id="transmission",
                         options=[
                             {"label": transmission, "value": transmission} for transmission  in TRANSMISSION
                         ], 
                         labelStyle={"color": "white"}
                     ),
                     width=7,
                 ),
             ],
             row=True,
             className= "pad-bottom"
         )
    
def _user_fueltype():
    return dbc.FormGroup(
              [
                  dbc.Label("Fuel Type", html_for="fuel_type", color="white"),
                  dcc.Dropdown(
                      id="fuel_type",
                      options=[
                          {"label": fueltype, "value": fueltype} for fueltype in FUEL_TYPE
                      ],
                      className= "dropdowns"
                  ),
              ], className= "pad-bottom"
          )
    
def _car_year():
    return dbc.FormGroup(
             [
                 dbc.Label("Car Year", html_for="car_year", width=2, color="white"),
                 dbc.Col(
                     daq.Slider(
                          id="car_year",
                          min=1996,
                          max=2021,
                          value=2019,
                          handleLabel={"showCurrentValue": True,"label": "Year"},
                          step=1,
                          size= 200,
                          className= "year-slider"
                     ),
                     width=5,
                 ),
             ],
             row=True,
             className= "pad-bottom"
         )
    
def _user_mpg():
    return dbc.FormGroup(
             [
                 dbc.Label("MPG", html_for="mpg", width=2, color="white"),
                 dbc.Col(
                     dbc.Input(
                         type="number", id="mpg", placeholder="Enter your car miles per gallon (mpg)"
                     ),
                     width=5,
                 ),
             ],
             row=True,
             className= "pad-bottom"
         )
    
def _car_tax():
    return dbc.FormGroup(
             [
                 dbc.Label("Tax", html_for="tax", width=2, color="white"),
                 dbc.Col(
                     dbc.Input(
                         type="number", id="tax", placeholder="Enter your car tax"
                     ),
                     width=5,
                 ),
             ],
             row=True,
             className= "pad-bottom"
         )

def _car_engine_class():
    return dbc.FormGroup(
              [
                  dbc.Label("Engine Class", html_for="engine_class", color="white"),
                  dcc.Dropdown(
                      id="engine_class",
                      options=[
                          {"label": engineclass, "value": engineclass} for engineclass in ENGINE_CLASS
                      ], 
                      className= "dropdowns"
                  ),
              ], className= "pad-bottom"
          )
    
def _car_modeltype():
    return dbc.FormGroup(
              [
                  dbc.Label("Model Type", html_for="model_type", color="white"),
                  dcc.Dropdown(
                      id="model_type",
                      options=[
                          {"label": modeltype, "value": modeltype} for modeltype in CAR_MODEL_TYPE
                      ],
                      className= "dropdowns"
                  ),
              ], className= "pad-bottom"
          )

def _create_submit_btn():
    return html.Div([
            dbc.Button("Submit", id="submit-btn", className="submit-btn", type="button"),
         ], style={"padding": "5%"}
    )

def _create_prediction_results():
    return html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalBody(
                    id="model-results",
                    style= {"padding": "3rem"}
                    ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close", className="ml-auto", n_clicks=0
                    )
                ),
            ],
            id="user_response",
            is_open=False,
            ),
        ]
    )


def create_car_info_page():
    return html.Div(
               [html.Br(), 
                dbc.Form(
                   [_user_mileage_info(), 
                    _user_car_transmission(), 
                    _user_fueltype(), 
                    _car_year(), 
                    _user_mpg(), 
                    _car_tax(), 
                    _car_engine_class(), 
                    _car_modeltype(),
                    _create_submit_btn(),
                    _create_prediction_results()
                    
               ]
           )], className="form-page"
    )

