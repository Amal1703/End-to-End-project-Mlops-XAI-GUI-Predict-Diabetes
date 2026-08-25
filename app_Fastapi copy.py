from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
import re
import subprocess
from pipeline.stage_6_predict_new_data import PredictNewData
import uvicorn



# Running uvicorn app_Fastapi:app --reload gives http://127.0.0.1:8000. Copy and paste it into a web browser


# initialize a fastapi app
app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key='Amal') # necessary to use session


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
                                    
            input_values = {}
            errors = {}
            feature_names, schema_columns = PredictNewData().get_input_column_X_train_and_schema()

            for col in feature_names:
                col_type = schema_columns[col]  # "int64" ou "float64"
                raw_value = form.get(col, "0").strip()

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

            request.session['last_values'] = dict(form) # the values from the user are saved in the predict fields
            
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
            prediction = PredictNewData().predict_new_data(**input_values)

            return templates.TemplateResponse('results.html', 
                                              {
                                                "request": request,
                                                "prediction": str(prediction)
                                                } )

    # GET, no button recognized -> display the empty form
    feature_names, schema_columns = PredictNewData().get_input_column_X_train_and_schema()
    return templates.TemplateResponse('predict.html',
                                       {   "request": request,
                                           "feature_names": feature_names,
                                           "values": request.session.get('last_values', {})
                                       } )
      
      
if __name__ == "__main__":
	# reload=True: reload the server when the Python code changes (for production, it must be removed)
	# uvicorn.run(host="0.0.0.0", port = 8080, reload=True)
 
	uvicorn.run(host="0.0.0.0", port = 8080)