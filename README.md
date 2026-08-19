# End-to-End-project-Mlops-XAI-GUI: Diabetes Prediction

### The dataset folder containing Predict Diabetes data was downloaded from: //www.kaggle.com/datasets/hasibur013/diabetes-dataset


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
python app.py
```
   - The notebook folder contains test.ipynb to test the pipeline in app.py
   - In params.yaml, you can change all the model parameters and you can add a different model 

### STEP 04 main.py ????


# Workflows

1. Create the necessary project folders and files 

2. Update config.yaml   # This file contains the application configuration: where to store files, where to download data, which folders to create and where to save models.

3. Update schema.yaml  # This file describes the data structure. It is used to validate that the input data matches the expected format.

4. Update params.yaml  # This file contains the model hyperparameters

5. Update the pipeline # This folder contains the pipeline. It coordinates the different stages (execution order). For example, for "Data Ingestion": it downloads the data, saves it, and extracts the files.

6. Update test.ipynb to test the pipeline

7. Update the app.py # Run the GUI

8. main.py ????
