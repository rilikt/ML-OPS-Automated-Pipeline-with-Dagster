from dagster import Definitions, io_manager
from .assets.load_data import raw_data, feature_engineering, preprocessing, train_model
from .io_managers.io_manager import local_csv_io, joblib_io_manager


defs = Definitions(
    assets=[raw_data, feature_engineering, preprocessing, train_model],
    resources={'io_manager': local_csv_io,
               'joblib_io_manager': joblib_io_manager,},
)