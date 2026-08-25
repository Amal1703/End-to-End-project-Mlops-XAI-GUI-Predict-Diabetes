import tkinter as tk
from tkinter import scrolledtext
import sys
from pipeline.stage_1_data_ingestion import DataIngestion
from pipeline.stage_2_data_validation import DataValidation
from pipeline.stage_3_data_transformation import DataTransformation
from pipeline.stage_4_model_trainer import ModelTrainer
from pipeline.stage_5_XAI_eXplainable_AI import XAI


def prompt_input(parent, title, message, default_value=""):

    result = {"value": None}

    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.geometry("350x150")
    dialog.resizable(False, False)

    # Center the window
    dialog.transient(parent)
    dialog.grab_set()

    # Message
    tk.Label(dialog, text=message, font=("Arial", 10)).pack(pady=10)

    # Entry field
    entry_var = tk.StringVar(value=default_value)
    entry = tk.Entry(dialog, textvariable=entry_var, width=40, font=("Arial", 10))
    entry.pack(pady=10)
    entry.focus_set()
    entry.select_range(0, tk.END)

    def on_ok():
        result["value"] = entry_var.get()
        dialog.destroy()

    def on_cancel():
        dialog.destroy()

    # Buttons
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="OK", command=on_ok, width=10, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Annuler", command=on_cancel, width=10, bg="#f44336", fg="white").pack(side=tk.LEFT, padx=10)

    # Enter shortcut
    entry.bind('<Return>', lambda e: on_ok())
    entry.bind('<Escape>', lambda e: on_cancel())

    parent.wait_window(dialog)

    return result["value"]

# ---------------------------------------------------------
# Represents a button that must first ASK the user for a value
# (via a dialog box), then call callback(value) with what was entered
# ---------------------------------------------------------
class AskForValue:
    def __init__(self, callback, type_="float"):
        """
    callback : function called with the entered value -> callback(value)
    type_ : "float" - controls which dialog box to use
        """
        self.callback = callback
        self.type_ = float


# ---------------------------------------------------------
# Redirect everything sent to print() to a text widget
# ---------------------------------------------------------
class ConsoleRedirector:
    def __init__(self, widget_text):
        self.widget_text = widget_text

    def write(self, message):
    # print() may call write() multiple times for a single line
    # (once per argument + separators): we insert it as-is,
    # WITHOUT adding a \n ourselves (print() already sends its own final \n).
        if message != "":
            self.widget_text.insert(tk.END, message)
            self.widget_text.see(tk.END)
        self.widget_text.update_idletasks()


    def flush(self):
        pass  # required by the sys.stdout interface, nothing to do here


# ---------------------------------------------------------
# Data structure: a menu tree (pipeline)
# Each node is either a sub-menu dict or a final result
# ---------------------------------------------------------
MENU = {
    "Data ingestion": {
        "Extract the zip file into the data directory":  lambda: DataIngestion().extract_zip_file(),
    },

    "Data validation": {
        "Visualize data and display the number of missing values": lambda: DataValidation().explore_data(),
        "Compare type and name columns of schema.yaml and data file": lambda: DataValidation().compare_type_and_name_columns(),
        "Visualize the distribution of a target variable (outcome)": lambda: DataValidation().visualize_distribution_target_variable(),
        "Plot histograms showing the distribution of all features for each category of a target variable": lambda: DataValidation().plot_histogram_each_column(),
    },

    "Data transformation": {
        "Filter methods": {
            "Correlation matrix": lambda: DataTransformation().correlation_matrix(),
            "Mutual information": lambda: DataTransformation().mutual_information(),
            "ANOVA F test": lambda: DataTransformation().ANOVA_F_test(),
        },

        "Embedded methods": {
            "Random forest": lambda: DataTransformation().random_forest(),
            "XGBoost": lambda: DataTransformation().XGBoost(),
        },

        "Show and save the 5 least important features to a file (based on XGBoost)": lambda: DataTransformation().save_least_important_features_to_file(),
    },

    "Model_trainer": {
        "Split data into training and test sets": AskForValue(
            callback=lambda test_size: ModelTrainer().split_data_train_test(test_size),
            type_="float",
        ),
        "Train and evaluate XGBoost by sequentially dropping features: first remove the 5 least important, then 4, then 3, until none remain.": lambda: ModelTrainer().XGBoost_evaluate_feature_drop(),
        "Train and evaluate SVM by dropping the features that were found by XGBoost to correspond to the lowest error rate": lambda: ModelTrainer().SVM_evaluate_feature_drop(),
        "Train and evaluate ANN by dropping the features that were found by XGBoost to correspond to the lowest error rate": lambda: ModelTrainer().ANN_evaluate_feature_drop(),
    },

    "XAI (Shap method using ANN model)": lambda: XAI().SHAP(),

}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stages of model training to predict diabetes (order of stages from top button to bottom)")
        self.geometry("400x400")

        # Navigation stack for handling the "Back" button
        self.historic = []

        self.frame = tk.Frame(self)
        self.frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Scrollable text area that will display print() output and results
        self.result_area = scrolledtext.ScrolledText(
            self, height=10, font=("Consolas", 10), fg="blue")

        self.result_area.pack(fill="both", expand=True, padx=10, pady=10)

        # Redirect all print() calls from the entire program to this text area
        sys.stdout = ConsoleRedirector(self.result_area)

        self.show_menu(MENU)


    def clear_frame(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

    def show_menu(self, current_menu):
        """Shows the buttons for the current menu"""
        self.clear_frame()

        for text, content in current_menu.items():
            btn = tk.Button(
                self.frame,
                text=text,
                width=25,
                height=3,
                wraplength=300,
                justify="center",
                #anchor="center",
                command=lambda c=content, t=text: self.click(c, t),
            )
            btn.pack(fill="x", pady=5)

        # Back button if not at the root menu
        if self.historic:
            back_button = tk.Button(
                self.frame, text="⬅ Back", width=25, command=self.back
            )
            back_button.pack(pady=15)

    def click(self, content, text):
        if isinstance(content, AskForValue):
            # First, prompt the user for a test size value using a popup dialog
            test_size_value = prompt_input(self,"Test size", "Enter any decimal value between 0 and 1", "0.2")
            test_size_value = float(test_size_value.replace(',', '.').strip()) if test_size_value.replace(',', '.').strip().replace('.', '').isdigit() else test_size_value # type(test_size_value) = str

            if (type(test_size_value) == float) and (0 < test_size_value < 1):
                print(f"--- {text} (Entered value : {test_size_value}) ---")
            try:
                content.callback(test_size_value)
            except Exception:
                print(f"ERROR (entered value: {test_size_value}), the test size must be a decimal between 0 and 1")

        if isinstance(content, dict):
            # It's a sub-menu: we move forward / we navigate deeper
            self.historic.append(content)
            self.show_menu(content)

        elif callable(content):
            # It's a function: we execute it now.
            print(f"--- Start : {text} ---")
            try:
                content()  # actual call, its print() output appears in real time.
                print(f"--- End : {text} ---\n")
            except Exception as e:
                print(f"Error : {e}")

        else:
            # final result (static text): we display it
            print(content)


    def back(self):
        self.historic.pop()
        previous_menu = self.historic[-1] if self.historic else MENU
        self.show_menu(previous_menu)


if __name__ == "__main__":
    app = App()
    app.mainloop()

