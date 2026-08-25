# This file is called a Dockerfile
# It describes how to build a Docker image for an application

# Select the base environment
FROM python:3.9-slim-bookworm

# Working directory inside the container (equivalent to cd /app_streamlit)
WORKDIR /app_streamlit

# Copy your project into the container
COPY . /app_streamlit 

 # Install the Python dependencies
RUN pip install -r requirements.txt 

# Run the application
CMD streamlit run app_streamlit.py --server.port=$PORT --server.address=0.0.0.0