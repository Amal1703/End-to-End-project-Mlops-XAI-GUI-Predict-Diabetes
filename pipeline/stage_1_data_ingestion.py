import zipfile
from utils.common import load_yaml_config
import os


class DataIngestion:

    def __init__(self, config_path: str = "yaml file/config.yaml"):
        self.config = load_yaml_config(config_path)

    def extract_zip_file(self) -> None:
        """
        Extract the zip file into the data directory
        """
        local_data_file = self.config["data_ingestion"]["local_data_file"]
        unzip_dir = self.config["data_ingestion"]["unzip_dir"]

        # Create the destination folder if it does not exist
        os.makedirs(unzip_dir, exist_ok=True)

        with zipfile.ZipFile(local_data_file, "r") as z:
            print("Name of files which are extracted:", z.namelist())  # List the files contained in the zip
            z.extractall(unzip_dir)
