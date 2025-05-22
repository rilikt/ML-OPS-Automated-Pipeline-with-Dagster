import pandas as pd
import dagster as dg
from dagster import file_relative_path, AssetOut, AssetExecutionContext
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
from typing import Tuple
from dagster import multi_asset, Output, Out


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

@multi_asset(
    outs={
        "train_data": AssetOut(io_manager_key='io_manager'),
        "test_data": AssetOut(io_manager_key='io_manager'),
    },
    deps=['feature_engineering'],
)
def train_test_sets(feature_engineering: pd.DataFrame):
    train_df, test_df = train_test_split(feature_engineering, test_size=0.2, random_state=42)
    yield Output(train_df, "train_data")
    yield Output(test_df, "test_data")


@dg.asset(io_manager_key='joblib_io_manager')
def preprocessing() -> Pipeline:
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

    regr = TransformedTargetRegressor(
        regressor=XGBRegressor(),
        func=np.log1p,
        inverse_func=np.expm1
    )

    pipe = Pipeline([('preprocessor', ct), ('regressor', regr)])
    return pipe

@dg.asset(deps=['train_data'], io_manager_key='joblib_io_manager')
def train_model(context: AssetExecutionContext, train_data: pd.DataFrame, preprocessing: Pipeline) -> Pipeline:
    with mlflow.start_run(run_name="dagster_training"):
        include = ['season', 'yr', 'hr_sin', 'hr_cos', 'mnth_sin','mnth_cos','weekday_sin', 'comfort_index',
                   'weekday_cos', 'workingday', 'holiday', 'weathersit', 'temp']
        data = train_data
        X = data[include].astype('float64')
        y = data['cnt']

        model = preprocessing
        model.fit(X, y)

    return model

@dg.asset(deps=['test_data', 'train_model'], io_manager_key='joblib_io_manager')
def evaluate_model(context: AssetExecutionContext, train_model: Pipeline, test_data: pd.DataFrame) -> Pipeline:

    include = ['season', 'yr', 'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos', 'weekday_sin', 'comfort_index',
               'weekday_cos', 'workingday', 'holiday', 'weathersit', 'temp']

    data = test_data
    X = data[include].astype('float64')
    y = data['cnt']

    input_example = X.iloc[:1]

    y_pred = train_model.predict(X)
    r2 = r2_score(y, y_pred)
    rmsle = np.sqrt(mean_squared_log_error(y, y_pred))
    context.log.info(f"Trained model R²: {r2}")
    context.log.info(f"Trained model Root Mean Squared log error: {rmsle}")

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("features_used", include)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("rmse", rmsle)
    mlflow.sklearn.log_model(train_model, artifact_path="model", input_example=input_example)

    return train_model