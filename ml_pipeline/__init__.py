from dagster import Definitions
from .assets.load_data import raw_data, feature_engineering, preprocessing, train_model, evaluate_model, train_test_sets
from .io_managers.io_manager import local_csv_io, joblib_io_manager
from .resources.mlflow_res import mlflow_tracking_resource
from .utils.utils import set_features_and_target

defs = Definitions(
    assets=[raw_data, feature_engineering, preprocessing, train_model, evaluate_model, train_test_sets],
    resources={'io_manager': local_csv_io,
               'joblib_io_manager': joblib_io_manager,
               'mlflow': mlflow_tracking_resource.configured({
                   'mlflow_tracking_uri': 'http://localhost:5000',
                   'experiment_name': 'dagster'}),
               },
)