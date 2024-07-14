import os
import sys
import yaml
import pickle
import pandas as pd

class Utils:

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    @staticmethod
    def add_src_to_path():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(dir_path)

    @staticmethod
    def get_project_root():
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    @staticmethod
    def load_parameters(parameters_directory):
        yaml_files = [f for f in os.listdir(parameters_directory) if f.endswith('.yml')]
        parameters = {}
        for yaml_file in yaml_files:
            with open(os.path.join(parameters_directory, yaml_file), 'r') as file:
                data = yaml.safe_load(file)
                key_name = f'parameters_{yaml_file.replace(".yml", "")}'
                parameters[key_name] = data
        return parameters

    @staticmethod
    def load_data(file_path):
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Formato de archivo no soportado. Use .csv, .parquet, o .xlsx")
        return data

    @staticmethod
    def save_data(data, path):
        if path.endswith('.parquet'):
            data.to_parquet(path)
        elif path.endswith('.csv'):
            data.to_csv(path, index=False)
        else:
            raise ValueError("Formato de archivo no soportado. Use .csv o .parquet")

    @staticmethod
    def load_pickle(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    @staticmethod
    def save_pickle(data, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
