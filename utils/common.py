import yaml
import os


def load_yaml_config(file_path):
    """
    Load a YAML configuration file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Configuration data loaded from the YAML file.

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    # Open and load the YAML file
    try:
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return config_data
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")
