from flask import Flask, render_template, request, session
import re
import subprocess
from pipeline.stage_6_predict_new_data import PredictNewData


# Running python app_Flask.py gives http://127.0.0.1:8080. Copy and paste it into a web browser


# initialize a flask app
app = Flask(__name__)
app.secret_key = 'Amal'  # necessary to use session


# route to display the home page
@app.route('/', methods=['GET'])
def homePage():
    return render_template("home.html")


@app.route('/predict_or_pipeline', methods=['POST', 'GET'])
def pipeline_predict():

    if request.method == 'POST':

        # case 1: "Pipeline" button was clicked
        if 'pipeline' in request.form:
            try:
                # runs the Tkinter interface in a separate process
                subprocess.Popen(['python', 'app_GUI.py'])
                return 'App GUI, which contains the model training stages, is launched'
            except Exception as e:
                return f"Error while starting the application: {e}", 500

        # case 2: "Predict" button was clicked
        elif 'predict' in request.form:
            input_values = {}
            errors = {}
            feature_names, schema_columns = PredictNewData().get_input_column_X_train_and_schema()

            for col in feature_names:
                col_type = schema_columns[col]  # "int64" ou "float64"
                raw_value = request.form.get(col, "0").strip()

                # validation based on column type
                if col_type == "int64":
                    if not raw_value.lstrip('-').isdigit() or int(raw_value) <= 0:
                        errors[col] = f"{col} value must be a positive integer"
                    else:
                        input_values[col] = int(raw_value)

                elif col_type == "float64":
                    if not re.match(r'^-?\d+(\.\d+)?$', raw_value) or float(raw_value) <= 0:
                        errors[col] = f"{col} value must be a positive float"
                    else:
                        input_values[col] = float(raw_value)

            session['last_values'] = request.form.to_dict() # the values from the user are saved in the predict fields

            if errors:
                return render_template(
                    'predict.html',
                    feature_names=feature_names,
                    errors=errors,
                    values=request.form
                )

            # all values are valid -> make the prediction
            prediction = PredictNewData().predict_new_data(**input_values)

            return render_template('results.html', prediction=str(prediction))

    # GET, no button recognized -> display the empty form
    feature_names, schema_columns = PredictNewData().get_input_column_X_train_and_schema()
    return render_template('predict.html', feature_names=feature_names, values=session.get('last_values', {}))


if __name__ == "__main__":
	# Without debug=True, Flask won't auto-reload the server or the templates.
	# To avoid having to restart manually after every change during development, enable
	# debug mode (for development only, never in production)
	#app.run(host="0.0.0.0", port = 8080, debug=True)

	app.run(host="0.0.0.0", port=8080)
