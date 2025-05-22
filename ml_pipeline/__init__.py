from dagster import Definitions, io_manager
from dagster_mlflow import mlflow_tracking
from .assets.load_data import raw_data, feature_engineering, preprocessing, train_model, evaluate_model, train_test_sets
from .io_managers.io_manager import local_csv_io, joblib_io_manager

defs = Definitions(
    assets=[raw_data, feature_engineering, preprocessing, train_model, evaluate_model, train_test_sets],
    resources={'io_manager': local_csv_io,
               'joblib_io_manager': joblib_io_manager,
               'mlflow': mlflow_tracking},
)