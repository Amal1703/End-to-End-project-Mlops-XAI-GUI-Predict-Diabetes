from utils.common import load_yaml_config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataValidation:

    def __init__(self, config_path: str = "yaml file/config.yaml", schema_filepath: str = "yaml file/schema.yaml"):
        self.config = load_yaml_config(config_path)
        self.schema = load_yaml_config(schema_filepath)
        self.df = pd.read_csv(self.config["data_validation"]["unzip_data_dir"])
        self.target = 'Outcome'

    # Compare type and name columns of schema and df
    def compare_type_and_name_columns(self) -> None:

            expected_schema = {**self.schema["COLUMNS"], **self.schema["TARGET_COLUMN"]}

            actual_schema = self.df.dtypes.astype(str).to_dict()
            actual_cols = set(actual_schema.keys())
            expected_cols = set(expected_schema.keys())

            missing_cols = expected_cols - actual_cols       # in schema but not in df (file data)
            extra_cols = actual_cols - expected_cols         # in df but not in schema
            common_cols = actual_cols & expected_cols

            type_mismatches = {
                col: {"expected in schema": expected_schema[col], "found in file data": actual_schema[col]}
                for col in common_cols
                if expected_schema[col] != actual_schema[col]}

            is_valid = not missing_cols and not extra_cols and not type_mismatches

            print("type and name of columns are valid:", is_valid)
            print("missing_columns (in schema but not in data file):", missing_cols)
            print("extra_columns (in data file but not in schema):", extra_cols)
            print("type_invalid:", type_mismatches)

    # Visualize data and display the number of missing values
    def explore_data (self) -> None: 
        print("Visualize data\n", self.df.head())
        print("rows, column:", self.df.shape)
        print("")
        print("Number of missing values per column\n", self.df.isnull().sum()) # Number of missing values per column
        print("\nData columns and their type")
        print(self.df.info()) # data columns and their type

    # Visualize the distribution of a target variable (outcome) 
    def visualize_distribution_target_variable (self) -> None: 
        sns.countplot(x=self.df[self.target], data=self.df)
        plt.title("Distribution of the target variable (outcome)")
        plt.show()

    # Plot histograms showing the distribution of all features for each category of a target variable
    def plot_histogram_each_column(self) -> None: 
        
        # Plot histograms showing the distribution of the feature for each category of a target variable
        def plotHistogram(values, label, feature, title, fig_num):
            sns.set_style("whitegrid")
            plotOne = sns.FacetGrid(values, hue=label, aspect=2)
            plotOne.map(sns.histplot, feature)
            plotOne.set(xlim=(0, values[feature].max()))
            plotOne.add_legend()
            plotOne.set_axis_labels(feature, 'Proportion')
            plotOne.fig.suptitle(f'Fig {fig_num}: {title}')
            plotOne.fig.canvas.manager.set_window_title(f'Fig {fig_num}')  # names the window "Fig 1", "Fig 2", etc.
        
        name_cols = self.df.select_dtypes(include="number").columns
        for i, col in enumerate(name_cols, start=1):
            plotHistogram(self.df, self.target, col, f'{col} (Blue = Healthy; Orange = Diabetes)', i)

        plt.show()  # shows ALL figures (fig1...fign) at once, at the end


# We can add a function to Boxplots for outlier detection (see test.py)            