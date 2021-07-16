import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_trich_components as dtc

__all__ = ["create_home_page", "create_navbar"]

def create_navbar():
        return dbc.NavbarSimple(
                    children=[
                       dbc.NavItem(dbc.NavLink("Explore the market", id="explore", href="/data-exploration")),
                       dbc.NavItem(dbc.NavLink("Predict a used car", id="predict", href="/predict-a-used-BMW-car"))
                   ],
                    brand="BMW Used Car Prediction",
                    brand_href="/",
                    color="success",
                    dark=True,
    )

def _create_welcome_content():
    return dbc.Jumbotron(
               [
                   html.H1("Welcome", className="display-3"),
                   html.P(
                         "Would you like to get a price estimate "
                         "of a used BMW car price?",
                         className="lead",
                     ),
        html.Hr(className="my-2"),
        html.P(
            "Predictive features include the car year, mileage, transmission, fuel type, "
            "tax, mpg, engine type, car model"
        ),
        html.P(dbc.Button("Start here", id="start-btn", color="primary", href="/predict-a-used-BMW-car"), className="lead"),
    ], className="jumbotron bg-dark text-white"
)
    
def _create_carosel():
    return dtc.Carousel([
           	html.Div(html.Img(src=f"./assets/images/{str(i).zfill(2)}.jpg", height="300px"), style={"padding": "2%"}) 
            for i in range(1, 11)
		],
        slides_to_scroll=1,
        swipe_to_slide=True,
        autoplay=True,
        speed=2000,
        variable_width=True,
        center_mode=True,
        responsive=[
               {
                 'breakpoint': 1024,
                 'settings': {
                   'slidesToShow': 3,
                   'slidesToScroll': 3,
                   'infinite': True,
                   'dots': True
                 }
               },
               {
                 'breakpoint': 600,
                 'settings': {
                   'slidesToShow': 2,
                   'slidesToScroll': 2,
                   'initialSlide': 2
                 }
               },
               {
                 'breakpoint': 480,
                 'settings': {
                   'slidesToShow': 1,
                   'slidesToScroll': 1
                 }
               }
             ]
	    )
    
def create_home_page():
  return html.Div([
                   html.Br(),
                   html.Br(),
                   _create_carosel(),
                   html.Br(),
                   _create_welcome_content()
             ])