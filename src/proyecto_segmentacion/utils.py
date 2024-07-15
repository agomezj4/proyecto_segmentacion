from typing import Tuple, Dict

import os
import sys
import yaml
import pickle
import logging
import pandas as pd
import numpy as np


class Utils:

    @staticmethod
    def setup_logging() -> logging.Logger:
        """
        Configura el logging para la aplicación.

        Returns
        -------
        logging.Logger
            El logger configurado para la aplicación.
        """
        import logging
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    @staticmethod
    def add_src_to_path() -> None:
        """
        Agrega la ruta del directorio 'src' al sys.path para facilitar las importaciones.

        Returns
        -------
        None
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        sys.path.append(project_root)

    @staticmethod
    def get_project_root() -> str:
        """
        Obtiene la ruta raíz del proyecto.

        Returns
        -------
        str
            Ruta raíz del proyecto.
        """
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    @staticmethod
    def load_parameters(parameters_directory: str) -> Dict[str, dict]:
        """
        Carga los archivos de parámetros en formato YAML desde un directorio específico.

        Parameters
        ----------
        parameters_directory : str
            Directorio donde se encuentran los archivos YAML.

        Returns
        -------
        Dict[str, dict]
            Diccionario con los parámetros cargados.
        """
        yaml_files = [f for f in os.listdir(parameters_directory) if f.endswith('.yml')]
        parameters = {}
        for yaml_file in yaml_files:
            with open(os.path.join(parameters_directory, yaml_file), 'r') as file:
                data = yaml.safe_load(file)
                key_name = f'parameters_{yaml_file.replace(".yml", "")}'
                parameters[key_name] = data
        return parameters

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo.

        Parameters
        ----------
        file_path : str
            Ruta del archivo a cargar. Puede ser .csv, .parquet, .xlsx o .xls.

        Returns
        -------
        pd.DataFrame
            DataFrame con los datos cargados.

        Raises
        ------
        ValueError
            Si el formato del archivo no es soportado.
        """
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
    def save_data(data: pd.DataFrame, path: str) -> None:
        """
        Guarda un DataFrame en un archivo.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a guardar.
        path : str
            Ruta del archivo donde se guardará el DataFrame. Puede ser .csv o .parquet.

        Raises
        ------
        ValueError
            Si el formato del archivo no es soportado.
        """
        if path.endswith('.parquet'):
            data.to_parquet(path)
        elif path.endswith('.csv'):
            data.to_csv(path, index=False)
        else:
            raise ValueError("Formato de archivo no soportado. Use .csv o .parquet")

    @staticmethod
    def load_pickle(file_path: str) -> object:
        """
        Carga un objeto desde un archivo pickle.

        Parameters
        ----------
        file_path : str
            Ruta del archivo pickle.

        Returns
        -------
        object
            Objeto cargado desde el archivo pickle.
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    @staticmethod
    def save_pickle(data: object, file_path: str) -> None:
        """
        Guarda un objeto en un archivo pickle.

        Parameters
        ----------
        data : object
            Objeto a guardar.
        file_path : str
            Ruta del archivo pickle donde se guardará el objeto.

        Returns
        -------
        None
        """
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def prepare_data(df: pd.DataFrame, exclude_columns: list) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara los datos para el entrenamiento eliminando temporalmente las columnas excluidas.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada.
        exclude_columns : list
            Lista de columnas a excluir.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            DataFrame con las columnas excluidas eliminadas y Serie con el ID del cliente.
        """
        cliente_id = df[exclude_columns[0]]
        df = df.drop(columns=exclude_columns)
        return df, cliente_id

    @staticmethod
    def join_clusters(cliente_id: pd.Series, labels: np.ndarray) -> pd.DataFrame:
        """
        Une los clusters al DataFrame original usando el ID del cliente.

        Parameters
        ----------
        cliente_id : pd.Series
            Serie con el ID del cliente.
        labels : np.ndarray
            Array con las etiquetas de los clusters.

        Returns
        -------
        pd.DataFrame
            DataFrame con el ID del cliente y los clusters asignados.
        """
        return pd.DataFrame({'cliente_id': cliente_id, 'cluster': labels})
