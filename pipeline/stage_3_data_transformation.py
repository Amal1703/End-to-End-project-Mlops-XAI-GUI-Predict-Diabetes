from utils.common import load_yaml_config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.feature_selection import f_classif  # ANOVA F-test
from sklearn.feature_selection import mutual_info_classif  # Mutual Information
import xgboost as xgb  # XGboost
from sklearn.ensemble import RandomForestClassifier  # Random Forest


class DataTransformation:

    def __init__(self, config_path: str = "yaml file/config.yaml", schema_filepath: str = "yaml file/schema.yaml"):
        self.config = load_yaml_config(config_path)
        self.schema = load_yaml_config(schema_filepath)
        self.df = pd.read_csv(self.config["data_validation"]["unzip_data_dir"])
        self.target = 'Outcome'
        self.X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]

# 1. Filter methods

    def correlation_matrix(self) -> None:
        plt.figure(figsize=(10, 8))
        corr = (self.df).corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def ANOVA_F_test(self) -> None:
        # Calculate F-scores and p-value
        f_scores, p_values = f_classif(self.X, self.y)

        # Sort by F-Score (from highest to lowest)
        results = np.argsort(f_scores)[::-1]

        # Display results
        for i in results:
            print(f"{self.X.columns[i]}: F-score = {f_scores[i]:.4f},\
                    p-value = {p_values[i]:.4f}")

    def mutual_information(self) -> None:
        # Calculate MI for all columns
        mi_scores = mutual_info_classif (self.X, self.y)

        # Sort by MI value (from highest to lowest)
        sorted_indices = np.argsort(mi_scores)[::-1]
        for i in sorted_indices:
            print(f"{self.df.drop(self.target, axis=1).columns[i]}: MI = {mi_scores[i]:.4f}")


# 2. Embedded methods

    def random_forest(self) -> None:
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(self.X, self.y)
        importance = pd.Series(rf.feature_importances_, index=self.X.columns).sort_values(ascending=False)
        print(importance)

    def XGBoost(self):
        xgb_model = xgb.XGBClassifier(n_estimators=200, random_state=42)
        xgb_model.fit(self.X, self.y)
        importance = pd.Series(xgb_model.feature_importances_, index=self.X.columns).sort_values(ascending=False)
        print(importance)
        return importance

    def save_least_important_features_to_file (self) -> None:
        # Based on the results from all methods:
        # 'Insulin', 'Pregnancies', 'SkinThickness', 'BloodPressure', and 'DiabetesPedigreeFunction' are the 5 least important features.
        # We will use the XGBoost results to save the list in the txt file.
        importance_XGBoost = DataTransformation().XGBoost()
        list_feature_drop = list(importance_XGBoost.keys()[3:])[::-1]
        print("5 least important features (according to XGBoost):", list_feature_drop)
        print("Other feature selection methods generally give the same result")

        # Save the list of the feature to drop in data_transformation
        os.makedirs(os.path.dirname(self.config["data_transformation"]["list_feature_drop_file"]) or ".", exist_ok=True)
        with open(self.config["data_transformation"]["list_feature_drop_file"], "w") as f:
            for element in list_feature_drop:
                f.write(str(element) + "\n")
