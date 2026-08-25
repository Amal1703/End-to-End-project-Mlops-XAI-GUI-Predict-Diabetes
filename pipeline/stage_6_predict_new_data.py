import joblib
from utils.common import load_yaml_config
import numpy as np
import pandas as pd


class PredictNewData: # (using ANN)

    def __init__(self, config_path: str = "yaml file/config.yaml", schema_filepath: str = "yaml file/schema.yaml"):
        self.config = load_yaml_config(config_path)
        self.schema = load_yaml_config(schema_filepath)
        self.model_ANN = joblib.load(self.config["model_trainer"]['model_path'])
        self.scaler_load = joblib.load(self.config["model_trainer"]['scaler_path'])
        
        self.df_train = pd.read_csv(self.config["model_trainer"]["train_data_ANN_path"])
        self.target = 'Outcome'
        self.X_train = self.df_train.drop(self.target, axis=1)
        self.feature_names = list(self.X_train.columns)
        
    # X_train_2 columns changes depending on which features are dropped (we don't know the arguments).
    def predict_new_data(self, **kwargs) -> int:
        # Compare the number of arguments received (len(kwargs)) to the expected number of columns (len(X_train.columns))
        if len(kwargs) != len(self.feature_names):
          raise ValueError(
            f"Incorrect number of values: {len(kwargs)} provided, "
            f"{len(self.feature_names)} expected ({list(self.feature_names)})"
        )    
        
        # DataFrame with the expected set of column names
        data = pd.DataFrame([kwargs], columns=self.feature_names)
        
        X_new_test_scaled = self.scaler_load.transform(data)
        y_pred_prob_new = self.model_ANN.predict(X_new_test_scaled, verbose=0)
        # Conversion with threshold at 0.5 (standard)
        y_pred_new = (y_pred_prob_new.flatten() > 0.5).astype(int)
        return int(y_pred_new[0])
    
    def get_input_column_X_train_and_schema (self):
        return self.feature_names, self.schema["COLUMNS"]