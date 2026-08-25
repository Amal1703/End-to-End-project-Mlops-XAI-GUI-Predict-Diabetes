
import streamlit as st
import io
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from datetime import datetime
import re
from pipeline.stage_1_data_ingestion import DataIngestion
from pipeline.stage_2_data_validation import DataValidation
from pipeline.stage_3_data_transformation import DataTransformation
from pipeline.stage_4_model_trainer import ModelTrainer
from pipeline.stage_5_XAI_eXplainable_AI import XAI
from pipeline.stage_6_predict_new_data import PredictNewData


# streamlit run app_streamlit.py


# Initialization
if "test_size" not in st.session_state:
    st.session_state.test_size = 0.2

if "show_dialog" not in st.session_state:
    st.session_state.show_dialog = False

if "dialog_action" not in st.session_state:
    st.session_state.dialog_action = False

if "log_output" not in st.session_state:
    st.session_state.log_output = ""


def _dialog_content():
    """Dialogue content"""
    st.write("### ⚙️ Configuration of test_size (between 0.01 and 0.99)")

    new_value = st.number_input(
        "Value of test_size",
        min_value=0.01,
        max_value=0.99,
        value=round(st.session_state.test_size, 2))

    col_ok, col_cancel = st.columns(2)

    with col_ok:
        if st.button("✅ Confirm", use_container_width=True):
            st.session_state.test_size = new_value
            st.session_state.show_dialog = False
            st.session_state.dialog_action = 'validated'
            st.success(f"✅ Test_size = {new_value:.2f}")
            st.rerun()
            return new_value

    with col_cancel:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_dialog = False
            st.session_state.dialog_action = 'cancelled'
            st.rerun()
            return None

    return None


def display_results(func, title=""):
    """Display the print() outputs of a function "func" and its matplotlib plots"""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        func()

    # Display of the generated figures
    fignums = plt.get_fignums()
    for num in fignums:
        fig = plt.figure(num)
        st.pyplot(fig)

    # Store the logs (results)
    timestamp = datetime.now().strftime("%H:%M:%S")

    if plt.get_fignums():
        entry =  (
            f"\n{'='*70}\n"
            f"[{timestamp}] {title}\n"
            f"{'='*70}\n"
            f"{buffer.getvalue()}"
            f"📊 {len(plt.get_fignums())} generated graph(s)\n"
            f"{'='*70}"
            f"\n✅ End of {title}\n"
            f"{'='*70}\n")
    else:
        entry =  (
            f"\n{'='*70}\n"
            f"[{timestamp}] {title}\n"
            f"{'='*70}\n"
            f"{buffer.getvalue()}"
            f"{'='*70}"
            f"\n✅ End of {title}\n"
            f"{'='*70}\n")
    st.session_state.log_output += entry


def render_toggle_stage_button(i) :
    """ Renders a Streamlit button for stage `i` that acts as a collapsible toggle

        The button label combines the stage's name with a directional arrow
        ("▲" if expanded, "▼" if collapsed), reflecting the current state
        stored in st.session_state[session_state[i]].

        Args:
        i (int): Index of the stage, used to look up its name, its
            session_state and its widget key from the
            corresponding lists (stage_name, session_state, key).
    """

    stage_name = ["📊 Data validation ", "🔧 Data transformation", "1: Filter methods",
                  "2: Embedded methods", "🧠 Model trainer", "🎯 Diabetes Prediction with ANN"]

    session_state = ["show_validation", "show_transformation", "show_Filter_methods",
                     "show_Embedded_methods", "show_Model_trainer", "show_predict"]

    key = ["btn2", "btn3", "btn_Filter_methods",
           "btn_Embedded_methods", "btn4", "btn6"]

    arrow = "▲" if st.session_state[session_state[i]] else "▼"
    if st.button(stage_name[i] +"  "+  f"{arrow}", use_container_width=True, key=key[i]):
        st.session_state[session_state[i]] = not st.session_state[session_state[i]]
        st.rerun()   # force an immediate rerun to sync the display upon this click


# Define the style of buttons
st.markdown("""
<style>
/* Target all buttons with keys starting with "btn" */
.st-key-btn1 button,
.st-key-btn2 button,
.st-key-btn3 button,
.st-key-btn4 button,
.st-key-btn5 button,
.st-key-btn6 button,{
    white-space: normal;  /* Allows text to wrap */
    overflow: visible;
    word-wrap: break-word;
    word-break: break-word;
    height: auto;  /* Auto height based on content */
    min-height: 100px;  /* Minimum height */
    width: 62mm;  /* Full width of container */
    padding: 20px;  /* Inner spacing */
    line-height: 1.4;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    box-sizing: border-box;
    font-size: 12px;  /* Adjust as needed */
}

/* Style for sub-buttons of data transformation */
.st-key-sub_buttons div.stButton > button {
    white-space: normal;
    overflow: visible;
    word-wrap: break-word;
    word-break: break-word;
    text-overflow: clip;
    width: auto;
    height: auto;  /* Auto height based on content */
    min-width: 30mm;
    min-height: 2em;
    font-size: 8px;
    word-wrap: break-word;
    line-height: 1.2;
    padding: 0.2em 0.4em;
}

</style>
""", unsafe_allow_html=True)


st.title("📋 Diabetes Prediction Pipeline")

# Define 6 buttons
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)


with col1:
    if st.button("📁 Data ingestion", use_container_width=True, key="btn1"):
        display_results(lambda: DataIngestion().extract_zip_file(), title="Data ingestion")

with col2:

    if "show_validation" not in st.session_state:
        st.session_state["show_validation"] = False

    render_toggle_stage_button(0)  # Data validation

    if st.session_state["show_validation"]:
        if st.button("1: Visualize data and display the number of missing value"):
            display_results(lambda: DataValidation().explore_data(), title="Visualize data and display the number of missing value")

        if st.button("2: Compare type and name columns of schema.yaml and data file"):
            display_results(lambda: DataValidation().compare_type_and_name_columns(), title="Compare type and name columns of schema.yaml and data file")

        if st.button("3: Visualize the distribution of a target variable (outcome)"):
            display_results(lambda: DataValidation().visualize_distribution_target_variable(), title="Visualize the distribution of a target variable (outcome)")

        if st.button("4: Plot histograms showing the distribution of all features for each category of a target variable"):
            display_results(lambda: DataValidation().plot_histogram_each_column(), title="Plot histograms showing the distribution of all features for each category of a target variable")

with col3:

    if "show_transformation" not in st.session_state:
        st.session_state["show_transformation"] = False

    render_toggle_stage_button(1)  # Data transformation

    if st.session_state["show_transformation"]:
        with st.container(key="sub_buttons"):

            # Define 3 buttons
            b1, b2 = st.columns(2, gap="small")
            b3 = st.columns(1)[0]

            with b1:
                if "show_Filter_methods" not in st.session_state:
                    st.session_state["show_Filter_methods"] = False

                render_toggle_stage_button(2)  # Filter methods

                if st.session_state["show_Filter_methods"]:

                    if st.button("1: Correlation matrix"):
                        display_results(lambda: DataTransformation().correlation_matrix(), title="Correlation matrix")

                    if st.button("2: Mutual information"):
                        display_results(lambda: DataTransformation().mutual_information(), title="Mutual information")

                    if st.button("3: ANOVA F test"):
                        display_results(lambda: DataTransformation().ANOVA_F_test(), title=" ANOVA F test")

            with b2:
                if "show_Embedded_methods" not in st.session_state:
                    st.session_state["show_Embedded_methods"] = False

                render_toggle_stage_button(3)  # Embedded methods

                if st.session_state["show_Embedded_methods"]:

                    if st.button("1: Random forest"):
                        display_results(lambda: DataTransformation().random_forest(), title="Random forest")

                    if st.button("2: XGBoost"):
                        display_results(lambda: DataTransformation().XGBoost(), title="XGBoost")

            with b3:
                if st.button("3: Show and save the 5 least important features to a file (based on XGBoost)"):
                    display_results(lambda: DataTransformation().save_least_important_features_to_file(), title="Show and save the 5 least important features to a file (based on XGBoost)")


with col4:

    if "show_Model_trainer" not in st.session_state:
        st.session_state["show_Model_trainer"] = False

    render_toggle_stage_button(4)

    if st.session_state["show_Model_trainer"]:
        if st.button("1: Split data into training and test sets"):
            st.session_state.show_dialog = True
            st.session_state.dialog_action = None
            st.rerun()

        # Open the dialog box
        if st.session_state.show_dialog == True:
            test_size = _dialog_content()

        # Display the logs if dialog is closed and validated
        elif st.session_state.dialog_action == 'validated':
            display_results(lambda: ModelTrainer().split_data_train_test(st.session_state.test_size),
                            title="Split data into training and test sets")
            st.session_state.dialog_action = None

        if st.button("2: Train and evaluate XGBoost by sequentially dropping features: first remove the 5 least important, then 4, then 3, until none remain"):
            display_results(lambda: ModelTrainer().XGBoost_evaluate_feature_drop(), title="Train and evaluate XGBoost by sequentially dropping features: first remove the 5 least important, then 4, then 3, until none remain")

        if st.button("3: Train and evaluate SVM by dropping the features that were found by XGBoost to correspond to the lowest error rate"):
            display_results(lambda: ModelTrainer().SVM_evaluate_feature_drop(), title="Train and evaluate SVM by dropping the features that were found by XGBoost to correspond to the lowest error rate")

        if st.button("4: Train and evaluate ANN by dropping the features that were found by XGBoost to correspond to the lowest error rate"):
            display_results(lambda: ModelTrainer().ANN_evaluate_feature_drop(), title="Train and evaluate ANN by dropping the features that were found by XGBoost to correspond to the lowest error rate")


with col5:
    if st.button("🔬 XAI (Shap method using ANN model)", use_container_width=True, key="btn5"):
        display_results(lambda: XAI().SHAP(), title="XAI (Shap method using ANN model)")


with col6:

    if "show_predict" not in st.session_state:
        st.session_state["show_predict"] = False

    render_toggle_stage_button(5)

    if st.session_state["show_predict"]:

        input_values = {}
        feature_names, schema_columns = PredictNewData().get_input_column_X_train_and_schema()

        for col in feature_names:
            col_type = schema_columns[col]  # "int64" ou "float64"
            raw_value = st.text_input(label=col, value="0", key=f"input_{col}")

            # validate the column type
            if col_type == "int64":
                if not (raw_value.strip().lstrip('-').isdigit()) or (int(raw_value) <= 0):
                    st.error(f"{col} value must be a positive integer")
                else:
                    input_values[col] = int(raw_value)

            elif col_type == "float64":
                if not re.match(r'^-?\d+(\.\d+)?$', raw_value.strip()) or (float(raw_value) <= 0):
                    st.error(f"{col} value must be a positive float")
                else:
                    input_values[col] = float(raw_value)

        if st.session_state["show_predict"]:
            if st.button("Diabetes Prediction using ANN"):
                display_results(lambda: print("Diabetes Prediction (1: diabetes; 0: no diabetes): ",
                                PredictNewData().predict_new_data(**input_values)),
                                title="Diabetes Prediction using ANN")

# === Display all logs in this window ===
st.text_area("Logs", st.session_state.log_output, height=500)

# Button to clear everything
if st.button("🗑️ Clear logs and plots"):
    st.session_state.log_output = ""
    st.session_state.figures = []
    st.rerun()
