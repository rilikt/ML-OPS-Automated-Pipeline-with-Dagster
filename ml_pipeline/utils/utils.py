import pandas as pd

def set_features_and_target(df: pd.DataFrame):
    include = ['season', 'yr', 'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos', 'weekday_sin', 'comfort_index',
               'weekday_cos', 'workingday', 'holiday', 'weathersit', 'temp']
    X = df[include].astype('float64')
    y = df['cnt']
    return X, y