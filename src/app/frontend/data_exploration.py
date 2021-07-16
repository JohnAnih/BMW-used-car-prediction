import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_loading_spinners as dls


COLUMNS = ['price', 'transmission', 'mileage', 'fueltype', 
           'tax', 'mpg', 'car_age', 'engine_class', 'model_type']

SPLIT_BY = ['transmission', 'fueltype', 'engine_class', 
            'model_type','car_age', 'nothing']

PLOT_TYPES = ["distribution plot", "violin plot"]

class DataExploration:
    def __init__(self):
        self._create_controls()        
    
    def _create_controls(self):
        self.controls = dbc.Card(
            [
             dbc.FormGroup(
                 [
                     dbc.Label("Features", className = "control-lables"),
                     dcc.Dropdown(
                         id="features",
                         options=[
                             {"label": col, "value": col} for col in COLUMNS
                         ],
                         value="price",
                     ),
                 ]
             ),
             html.Br(),
             dbc.FormGroup(
                 [
                     dbc.Label("Split feature by: ", className = "control-lables"),
                     dcc.Dropdown(
                         id="hue",
                         options=[
                             {"label": col, "value": col} for col in SPLIT_BY
                         ],
                         value="engine_class",
                     ),
                 ]
             ),
             html.Br(),
             dbc.FormGroup(
                 [
                     dbc.Label("Plot type", className = "control-lables"),
                     dcc.Dropdown(
                         id="plot_type",
                         options=[
                             {"label": col, "value": col} for col in PLOT_TYPES
                         ],
                         value="violin plot",
                     ),
                 ]
             ),
         ],
        body=True,
        className = "controls"
      )

    @property
    def layout(self):
        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(self.controls, md=4),
                        dbc.Col(
                            dls.ClimbingBox(dcc.Graph(id="plot_data"), 
                                            color="#28a745",
                                            speed_multiplier=2,
                                            fullscreen=True), md=8),
         
                    ],
                    align="center",
                ),
            ],
            fluid=True,
        )