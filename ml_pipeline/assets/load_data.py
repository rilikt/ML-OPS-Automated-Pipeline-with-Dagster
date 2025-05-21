import pandas as pd
import dagster as dg
from dagster import file_relative_path, AssetOut
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn


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
    df = raw_data.copy()
    df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24) # transform time data (cyclical encoding to continuous circular range)
    df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24)
    df['mnth_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
    df['mnth_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['comfort_index'] = df['temp'] * (1 - df['hum'])

    return df

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

@dg.asset(deps=['preprocessing'], io_manager_key='joblib_io_manager')
def train_model(context, feature_engineering: pd.DataFrame, preprocessing: ColumnTransformer) -> Pipeline:
    include = ['season', 'yr', 'hr_sin', 'hr_cos', 'mnth_sin','mnth_cos','weekday_sin', 'comfort_index','weekday_cos', 'workingday', 'holiday', 'weathersit', 'temp']
    data = feature_engineering
    X = data[include].astype('float64')
    y = data['cnt']

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
    input_example = X_train.iloc[:1]

    regr = TransformedTargetRegressor(
        regressor=XGBRegressor(),
        func=np.log1p,
        inverse_func=np.expm1
    )

    pipe = Pipeline([('preprocessor', preprocessing), ('regressor', regr)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    msle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    context.log.info(f"Trained model R²: {r2}")
    context.log.info(f"Trained model Root Mean Squared log error: {msle}")

    return pipe