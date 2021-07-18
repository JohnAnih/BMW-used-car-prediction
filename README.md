# BMW-used-car-prediction

This project was developed to predict the price for used BMW cars. The project was motivated from the DataCamp certification excercise.

Predictive features includes: The car model, the year of the car, mpg, car mileage and more.

# Folder Structure:

The project folder structure is described below:

```bash
├── data
│   ├── bmw.csv
├── notebooks
│   ├── eda.ipynb
    ├── experiment_anlysis.ipynb
    ├── spotcheck_algoirthms.ipynb
    ├── tune_algoirthms.ipynb
├── src
│   ├── app
│   │   ├── assets
│   │   ├── backend
│   │   │    ├── __init__.py
│   │   │    ├── data_model.py
│   │   │    ├── plotting.py
│   │   ├── frontend
│   │   │    ├── __init__.py
│   │   │    ├── data_exploration.py
│   │   │    ├── home_page.py
│   │   │    ├── prediction.py
│   │   │    ├── utils.py
│   │   ├── app.py
│   │   ├── __init__.py
│   ├── models
│   │   ├── model_store
│   │   │    ├── model.pkl
│   │   ├── __init__.py
│   │   ├── base_estimator.py
│   │   ├── compare_models.py
│   │   ├── estimators.py
│   │   ├── onehotcoder.py
│   │   ├── preprocessing.py
│   │   ├── settings.py
│   │   ├── tune_models.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── prepare_data.py
│   └── __init__.py
├── README.md
├── runtime.txt
├── requirements.txt
└── .gitignore
```

> The data used for this project is available in the data folder.
>
> The notebook files are available in the notebook folder, the notebook files contains the exploratory data analysis, model training and comparison, project summaries and thought process.
>
> The src directory contains the app folder, the data preprocessing pipeline and the model training pipeline
>
> The app utlizes the Dash library to develop both the frontend and backend

To see the live application demo please [click here](https://bmw-car-prediction.herokuapp.com/)
