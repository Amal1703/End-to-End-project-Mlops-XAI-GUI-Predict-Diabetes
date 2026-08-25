# End-to-End-project-Mlops-XAI-GUI: Diabetes Prediction (classification problem)

In this project, I have created 4 applications: GUI, Flask, Streamlit and Fastapi. To make these applications work properly, we need to follow this 6-stage pipeline:
   - data ingestion
   - data validation 
   - data transformation (in this stage, you will have many filter and embedded methods, you can show the results of all those methods, but to have the results of next stages, it is necessary to first obtain results of XGboost (embedded method) and then show and save the 5 least important features to a file)
   - model trainer (in this stage, it is necessary to first split data into training and test sets, second train XGBoost model and then train ANN model. In training XGBoost model, we will create 6 models by sequentially dropping features: first remove the 5 least important, then 4, then 3, until none remain and the model by dropping feature give the lowest error. Finally, train the ANN using those features. You can train an SVM model and show its results)
   - XAI 
   - diabetes prediction


### The dataset folder containing Predict Diabetes data was downloaded from: //www.kaggle.com/datasets/hasibur013/diabetes-dataset

## The dataset contains the following features:
 - Pregnancies (Integer): Number of times the patient has been pregnant.
 - Glucose (Integer): Plasma glucose concentration after a 2-hour oral glucose tolerance test.
 - BloodPressure (Integer): Diastolic blood pressure (mm Hg).
 - SkinThickness (Integer): Triceps skinfold thickness (mm).
 - Insulin (Integer): 2-hour serum insulin (mu U/ml).
 - BMI (Float): Body mass index (weight in kg/(height in m)^2).
 - DiabetesPedigreeFunction (Float): A function that represents the patient’s diabetes pedigree (i.e., likelihood of diabetes based on family history).
 - Age (Integer): Age of the patient (years).
 - Outcome (Binary/int): Binary outcome (0 or 1) where 1 indicates the presence of diabetes and 0 indicates the absence.


# How to run this project?
### STEPS:

### STEP 01 - Clone the repository
```bash, cmd
git clone https://github.com/Amal1703/End-to-End-project-Mlops-XAI-GUI-Predict-Diabetes.git
```

### STEP 02 - install the requirements
```bash, cmd
pip install -r requirements.txt
```

### STEP 03 - run the GUI (run the pipeline)
```bash, cmd
python app_GUI.py
```
   - The notebook folder contains test.ipynb to test the pipeline in app.py
   - In params.yaml, you can change all the model parameters and you can add a different model 

### STEP 04 - run the flask app 
```bash, cmd
python app_Flask.py
```
   - Running python app_Flask.py gives http://127.0.0.1:8080. Copy and paste it into a web browser

   - The web app contains 2 buttons:
       - 1st: "Pipeline" – when you click on it, it returns app_GUI.py
       - 2nd: "Predict" – you need to fill in some fields to predict diabetes

### STEP 05 - run the streamlit app  
```bash, cmd
streamlit run app_streamlit.py
```
   - The app contains 6 buttons of all the stages
   - In params.yaml, you can change all the model parameters and you can add a different model 

### STEP 06 - run the fastapi app 
```bash, cmd
python app_Fastapi.py
```
   - Running uvicorn app_Fastapi:app --reload gives http://127.0.0.1:8000. Copy and paste it into a web browser

   - The web app contains 2 buttons:
       - 1st: "Pipeline" – when you click on it, it returns app_GUI.py
       - 2nd: "Predict" – you need to fill in some fields to predict diabetes


# Workflows

1. Create the necessary project folders and files 

2. Update config.yaml   # This file contains the application configuration: where to store files, where to download data, which folders to create and where to save models.

3. Update schema.yaml  # This file describes the data structure. It is used to validate that the input data matches the expected format.

4. Update params.yaml  # This file contains the model hyperparameters

5. Update the pipeline # This folder contains the pipeline. It coordinates the different stages (execution order). For example, for "Data Ingestion": it downloads the data, saves it, and extracts the files.

6. Update test.ipynb to test the pipeline

7. Update the app_GUI.py # Run the GUI

8. Update the app_Flask.py 

9. Update the app_streamlit.py 

10. Update the app_Fastapi.py 

11. Deploying a Streamlit app on Google Cloud Console:
    - Create a Dockerfile 
    - Create a .dockerignore file to exclude other apps and unnecessary folders for the Docker image
    - In requirements.txt, comment out the following libraries (as they are not needed for deployment): notebook, flask, fastapi, python-multipart, itsdangerous, lime.

12. Create and fill in the CI/CD file, which is the main.yaml file in (.github\workflows). Add secrets to GitHub (Settings → Secrets and variables → Actions → click the "New repository secret" button and add the names and their secrets).
   - Example: click on New repository secret: Name GCP_REGION and secret europe-west9

For secrets.GCP_SA_KEY: you need to execute these 4 commands in order:

 - gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="serviceAccount:github-deployer@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/run.admin"

 - gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="serviceAccount:github-deployer@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/artifactregistry.writer"

 - gcloud projects add-iam-policy-binding ${PROJECT_ID} --member="serviceAccount:github-deployer${PROJECT_ID}.iam.gserviceaccount.com" --role="roles/iam.serviceAccountUser"

 - gcloud iam service-accounts keys create key.json --iam-account=github-deployer@${PROJECT_ID}.iam.gserviceaccount.com

=> key.json will be created in the folder. We have added key.json to .gitignore and .dockerignore

=> On GitHub repo, go to Settings → Secrets and variables → Actions → click the "New repository secret" button. In the Name field, enter exactly GCP_SA_KEY. In the Secret field, paste the copied JSON content. Click "Add secret"


### Deploying a Streamlit application on Google Cloud Console

1. Create an account on Google Cloud Console

2. Create a project on Google Cloud Console (here we retrieve the PROJECT_ID variable)

3. Install Google Cloud SDK: Follow the instructions at: https://docs.cloud.google.com/sdk/docs/install-sdk?hl=en:
       - To verify that it is installed correctly, run gcloud --version in the terminal

4. Create a Docker repository in artifact registry (search for "artifact registry" in the console). Here we define the following variables:
       - REGION : The location for your repository
       - ARTIFACT_REGISTRY_NAME : The name for the artifact repository

5. Build the Docker image in the Artifact Registry repository:

       - gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_NAME}/${IMAGE_NAME} 
       - IMAGE_NAME : This is the name you choose for your Docker image

6. Deploy the image to Google Cloud Run: 
      - gcloud run deploy ${SERVICE_NAME} `
        --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_NAME}/${IMAGE_NAME}:latest `
        --platform managed `
        --max-instances=1 `
        --region ${REGION} `
        --allow-unauthenticated

      - SERVICE_NAME : This the name you choose for your Cloud Run service. It identifies your deployed service in the Google Cloud Console and appears in the generated URL


7. After a successful deployment (no errors), the deployment command will output the public URL of your service directly in the terminal. It will look like: https://${SERVICE_NAME} -xxxxx-${REGION}.a.run.app
You can view the URL and logs in the Google Cloud Console
