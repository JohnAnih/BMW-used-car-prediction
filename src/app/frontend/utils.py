from collections import OrderedDict

import dash_bootstrap_components as dbc
import dash_html_components as html


def get_results(mileage: int, transmission: str, 
                fuel_type: str, car_year: int, mpg: int, 
                tax: int, engine_class: str, model_type: str, prediction: str):

    features = OrderedDict({
              "Mileage:": mileage, 
              "Transmission:": transmission, 
              "Fuel Type:": fuel_type,
              "Car Year:": car_year,
              "MPG:": mpg, 
              "Tax:": tax, 
              "Engine Class:": engine_class,
              "Car Model:": model_type,
         })
    
    results = [html.H3("You entered the following details:", 
                       className="h3-additional-style", style={"margin-bottom": "2%"}),
               
    ]

    for feature, user_input in features.items():
        results.append(html.B(feature),)
        results.append(html.P(f"{user_input}"),)
    results.append(dbc.Badge(f"Estimated price: ${prediction}", className="price-style"),)
    
    return results

def show_errror_msg():
    return "You must fill in all questions to make a predictions"