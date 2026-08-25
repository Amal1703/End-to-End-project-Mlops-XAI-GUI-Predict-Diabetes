from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import create_model, Field, ValidationError
import re
import subprocess
from pipeline.stage_6_predict_new_data import PredictNewData
import uvicorn


# Running uvicorn app_Fastapi:app --reload gives http://127.0.0.1:8000. Copy and paste it into a web browser


def validated_data():
    feature_names, schema_columns = PredictNewData().get_input_column_X_train_and_schema()
    fields = {}
    for col, col_type in schema_columns.items():
        if col_type == "int64":
            fields[col] = (int, Field(default=0, gt=0))
        elif col_type == "float64":
            fields[col] = (float, Field(default=0, gt=0)) 
    model = create_model("PatientData", **fields)
    return model, feature_names

validated_data_model, FEATURE_NAMES = validated_data()

# initialize a fastapi app
app = FastAPI()

# access the templates folder
templates = Jinja2Templates(directory="templates")

# access the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get('/')  
def homePage(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.api_route('/predict_or_pipeline', methods=['GET', 'POST'])
async def pipeline_predict(request: Request):
    
    form = await request.form()
    
    if request.method == 'POST':
        

        # case 1: "Pipeline" button was clicked
        if 'pipeline' in form:
            try:
                # runs the Tkinter interface in a separate process
                subprocess.Popen(['python', 'app_GUI.py'])
                return 'App GUI, which contains the model training stages, is launched'
            except Exception as e:
                return f"Error while starting the application: {e}", 500

        # case 2: "Predict" button was clicked
        elif 'predict' in form:                           
            feature_names = FEATURE_NAMES
            errors = {}

            try:

               form_dict = dict(form)
               model_fields = set(validated_data_model.model_fields.keys())
               # only keep the fields that the model expects ('predict' excluded)
               filtered_data = {k: v for k, v in form_dict.items() if k in model_fields}
               
              # check whether filtered_data is empty
               if not filtered_data:
                 # add default values or return an error
                 return templates.TemplateResponse("predict.html", {
                     "request": request,
                     "feature_names": feature_names,
                     "errors": errors,
                     "values": form_dict
                 })

               # validation based on column type
               data = validated_data_model(**filtered_data)

            except ValidationError as e:
                # convert Pydantic validation errors to a {field: message} dictionary
                for err in e.errors():
                    field = err["loc"][0]
                    errors[field] = err["msg"]

            if errors:
                return templates.TemplateResponse(
                "predict.html",
                        {
                            "request": request,
                            "feature_names": feature_names,
                            "errors": errors,
                            "values": form
                        }
                    )

            # all values are valid -> make the prediction  
            input_values =  {k: v for k, v in data.model_dump().items() if k in feature_names}

            prediction = PredictNewData().predict_new_data(**input_values)

            return templates.TemplateResponse('results.html', 
                                                {
                                                "request": request,
                                                "prediction": (prediction)
                                                } )

    else:
    # GET, no button recognized -> display the empty form
        feature_names = FEATURE_NAMES
        return templates.TemplateResponse('predict.html',
                                        {   "request": request,
                                            "feature_names": feature_names,
                                            "values": 0
                                        } )


if __name__ == "__main__":
	# reload=True: reload the server when the Python code changes (for production, it must be removed)
	uvicorn.run(host="0.0.0.0", port = 8080, reload=True)

	# uvicorn.run(host="0.0.0.0", port = 8080)
