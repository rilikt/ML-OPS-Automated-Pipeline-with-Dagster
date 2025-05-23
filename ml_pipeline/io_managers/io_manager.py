from dagster import IOManager, io_manager
import pandas as pd
import os
import joblib

class LocalCSVIOManager(IOManager):
    def handle_output(self, context, obj):
        output_name = context.asset_key.path[-1] # gets the current asset name
        path = f"outputs/{output_name}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        obj.to_csv(path, index=False)
        context.log.info(f"Saved output to {path}")

    def load_input(self, context):
        input_name = context.asset_key.path[-1]
        if input_name == "raw_data":
            path = "data/hour.csv"
        else:
            path = f"outputs/{input_name}.csv"
        context.log.info(f"Loading input from {path}")
        return pd.read_csv(path)

@io_manager
def local_csv_io(_):
    return LocalCSVIOManager()



class JoblibIOManager(IOManager):
    def _get_path(self, context):
        asset_id = context.asset_key.path[-1]
        return f"outputs/{asset_id}.joblib"

    def handle_output(self, context, obj):
        path = self._get_path(context)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(obj, path)

    def load_input(self, context):
        path = self._get_path(context)
        return joblib.load(path)

@io_manager
def joblib_io_manager(_):
    return JoblibIOManager()
