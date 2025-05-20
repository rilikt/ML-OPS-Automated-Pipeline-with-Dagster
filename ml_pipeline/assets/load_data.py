import pandas as pd
import dagster as dg
from dagster import file_relative_path, AssetOut
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path

@dg.asset
def raw_data() -> pd.DataFrame:
    """
    Reads csv data from local csv file
    Args:
        None
    Returns:
        pd.DataFrame
    """
    return pd.read_csv("data/hour.csv")

@dg.asset(deps=['raw_data'])
def feature_engineering(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering function for time based data
    Args:
        raw_data (pd.DataFrame), our raw input data
    Returns:
        pd.DataFrame and stores .csv with new features
    """
    raw_data['hr_sin'] = np.sin(2 * np.pi * raw_data['hr'] / 24) # transform time data (cyclical encoding to continuous circular range)
    raw_data['hr_cos'] = np.cos(2 * np.pi * raw_data['hr'] / 24)
    raw_data['mnth_sin'] = np.sin(2 * np.pi * raw_data['mnth'] / 12)
    raw_data['mnth_cos'] = np.cos(2 * np.pi * raw_data['mnth'] / 12)
    raw_data['weekday_sin'] = np.sin(2 * np.pi * raw_data['weekday'] / 7)
    raw_data['weekday_cos'] = np.cos(2 * np.pi * raw_data['weekday'] / 7)
    raw_data['comfort_index'] = raw_data['temp'] * (1 - raw_data['hum'])

    return raw_data

@dg.asset(deps=['feature_engineering'], io_manager_key='joblib_io_manager')
def preprocessing(feature_engineering: pd.DataFrame) -> ColumnTransformer:
    categorial_features = ['season', 'weathersit', 'yr']
    numerical_features = ['temp', 'comfort_index']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    ct = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorial_features),
        ],
        remainder = 'passthrough'
    )

    return ct

@dg.asset(deps=['preprocessing'])
def train_model(feature_engineering: pd.DataFrame, preprocessing: ColumnTransformer) -> None:
    include = ['season', 'yr', 'hr_sin', 'hr_cos', 'mnth_sin','mnth_cos','weekday_sin', 'comfort_index','weekday_cos', 'workingday', 'holiday', 'weathersit', 'temp']
    data = feature_engineering
    X = data[include]
    y = data['cnt']
    X = X.astype({col: 'float64' for col in X.select_dtypes(include='int').columns}) #needed?

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
    regr = TransformedTargetRegressor( #log to even out the data skewness
        regressor=LinearRegression(),
        func=np.log1p,
        inverse_func=np.expm1
    )

    pipe = Pipeline([('preprocessor', preprocessing), ('regressor', regr)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("r2_score", r2_score(y_test, y_pred))
