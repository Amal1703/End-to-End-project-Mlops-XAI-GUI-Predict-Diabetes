import warnings
import joblib
import matplotlib.pyplot as plt
from utils.common import load_yaml_config
import pandas as pd
# XAI
import shap


warnings.filterwarnings('ignore', category=UserWarning)


class XAI:

    def __init__(self, config_path: str = "yaml file/config.yaml"):
        self.config = load_yaml_config(config_path)
        self.model_ANN = joblib.load(self.config["model_trainer"]['model_path'])
        self.scaler_load = joblib.load(self.config["model_trainer"]['scaler_path'])

        self.df_train = pd.read_csv(self.config["model_trainer"]["train_data_ANN_path"])
        self.df_test = pd.read_csv(self.config["model_trainer"]["test_data_path"])
        self.target = 'Outcome'

        self.X_test = self.df_test.drop(self.target, axis=1)
        self.X_train = self.df_train.drop(self.target, axis=1)

        self.X_test_scaled = self.scaler_load.transform(self.X_test[self.X_train.columns])
        self.X_train_scaled = self.scaler_load.transform(self.X_train)

        self.feature_names = list(self.X_train.columns)

    def SHAP(self) -> None:

        # Create the explainer
        explainer = shap.Explainer(lambda x: self.model_ANN.predict(x, verbose=0), self.X_train_scaled)

        # Compute the SHAP values (for class 1, the model's native output)
        shap_values = explainer(self.X_test_scaled, silent=True)

        plt.figure()
        plt.title("Fig 1: SHAP - Importance of features (class 1: diabetic)")
        plt.tight_layout()

        # --- Summary plot for class 1  ---
        shap.summary_plot(shap_values,
                            self.X_test_scaled,
                            plot_type="bar",
                            feature_names=self.feature_names,
                            show=False)

        plt.show()

        # --- Summary plot for class 0 (just with the opposite sign) ---
        shap_values_class0 = shap.Explanation(values=-shap_values.values,  # opposite sign
                                                base_values=1 - shap_values.base_values,  # base_value is also inverted
                                                data=shap_values.data,
                                                feature_names=self.feature_names)

        plt.figure()
        plt.title("Fig 2: SHAP - Importance of features (class 0: non-diabetic)")
        plt.tight_layout()

        shap.summary_plot(shap_values_class0,
                            self.X_test_scaled,
                            plot_type="bar",
                            feature_names=self.feature_names,
                            show=False)

        plt.show()
# we can add Lime method (see test.py)