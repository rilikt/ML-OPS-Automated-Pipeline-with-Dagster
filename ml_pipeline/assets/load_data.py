import pandas as pd
import dagster as dg
from dagster import AssetOut
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from dagster import multi_asset, Output
from dagster import AssetExecutionContext
from ..utils.utils import set_features_and_target
import seaborn as sns
import matplotlib.pyplot as plt


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
    """
    Splitting the data into train and test sets
    Args:
        feature_engineering (pd.DataFrame), our feature engineering data
    Returns:
        Two dataframes with train and test sets
    """
    train_df, test_df = train_test_split(feature_engineering, test_size=0.2, random_state=42)
    yield Output(train_df, "train_data")
    yield Output(test_df, "test_data")


@dg.asset(io_manager_key='joblib_io_manager')
def preprocessing() -> Pipeline:
    """
    Preprocessing the data and prepare it for machine learning
    Args:
        features from our dataset
    Returns:
        processed data optimized for training
    """
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

@dg.asset(deps=['train_data'], io_manager_key='joblib_io_manager', required_resource_keys={'mlflow'})
def train_model(context: AssetExecutionContext, train_data: pd.DataFrame, preprocessing: Pipeline) -> Pipeline:
    """
    Train a machine learning model
    Args:
        our training data, and preprocessing pipeline
    Returns:
        a fitted model
    """
    mlflow = context.resources.mlflow

    with mlflow.start_run(run_name="dagster_training"):
        X, y = set_features_and_target(train_data)
        model = preprocessing
        model.fit(X, y)

    return model

@dg.asset(deps=['test_data', 'train_model'], io_manager_key='joblib_io_manager', required_resource_keys={'mlflow'})
def evaluate_model(context: AssetExecutionContext, train_model: Pipeline, test_data: pd.DataFrame) -> Pipeline:
    """
    Evaluate a machine learning model
    Args:
        a fitted model and test data set
    Returns:
        a fitted model
    """
    X, y = set_features_and_target(test_data)
    mlflow = context.resources.mlflow

    input_example = X.iloc[:1]
    with mlflow.start_run(run_name="dagster_evaluation"):
        y_pred = train_model.predict(X)
        r2 = r2_score(y, y_pred)
        rmsle = np.sqrt(mean_squared_log_error(y, y_pred))
        context.log.info(f"Trained model R²: {r2}")
        context.log.info(f"Trained model Root Mean Squared log error: {rmsle}")

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmsle)
        mlflow.sklearn.log_model(train_model, artifact_path="model", input_example=input_example)

        residuals = y - y_pred
        sns.scatterplot(x=y_pred, y=residuals, alpha=.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Count')
        plt.ylabel('Predicted')
        plt.title('Xgboost Regressor')
        plt.savefig('prediction.png')
        mlflow.log_artifact('prediction.png')

    return train_model
