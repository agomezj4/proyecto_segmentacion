from typing import Dict, Any

import pandas as pd
import unicodedata
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class pipelie_raw:

    # 1. Validar tag de las fuentes
    def validate_tags_pd(df: pd.DataFrame, tag_dict: pd.DataFrame) -> pd.DataFrame:
        """
        Valida que el número de tags identificados en source como raw sea igual al
        número de columnas en el dataframe utilizando Pandas.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe que contiene los datos a validar
        tag_dict : pd.DataFrame
            Diccionario de etiquetas

        Returns
        -------
        pd.DataFrame
            Si las tags están validadas, retorna el DataFrame original. En otros casos,
            solo emite un log
        """

        # Registra un mensaje de información indicando el inicio del proceso de validación de tags
        logger.info("Iniciando la validación de tags...")

        # Calcula la cantidad de tags identificados como 'raw' en el tag dictionary
        len_raw = tag_dict[tag_dict["source"] == "raw"].shape[0]

        # Comprueba si el número de tags es mayor que el número de columnas en el DataFrame
        if len_raw != df.shape[1]:
            raise ValueError(
                "Tags faltantes en el dataframe" if len_raw > df.shape[1] else "Tags faltantes en el tag dictionary")

        # Si el número de tags es igual al número de columnas, emite un registro informativo y retorna el DataFrame 'df'
        logger.info("Tags validados!")

        return df

    # 2. Validar tipos de datos
    def validate_dtypes_pd(df: pd.DataFrame, tag_dict: pd.DataFrame) -> pd.DataFrame:
        """
        Revisa que la tipología de los datos en el dataframe sea la misma estipulada
        en el tag_dictionary.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe que se quiere validar
        tag_dict : pd.DataFrame
            Diccionario de etiquetas

        Returns
        -------
        pd.DataFrame
            Si las tipologías de datos están validadas, retorna el DataFrame original.
            En otros casos, solo emite un log
        """

        logger.info("Iniciando la validación de tipos de datos...")

        # Filtra las filas del tag dictionary donde "source" es "raw"
        tag_dict_raw = tag_dict[tag_dict["source"] == "raw"]

        # Crear un diccionario de los tipos de datos esperados para cada columna, normalizando a minúsculas y eliminando espacios
        expected_types = {row['tag']: row['data_type'].strip().lower() for row in
                          tag_dict_raw.to_dict(orient='records')}

        # Genera una lista de problemas con las diferencias entre los tipos de datos en el DataFrame y el tag dictionary
        problems = [
            f"{col} is {df[col].dtype} but should be {expected_types[col]}"
            for col in df.columns if col in expected_types and str(df[col].dtype).strip().lower() != expected_types[col]
        ]

        # Comprueba si se encontraron problemas y lanza una excepción TypeError si es así
        if problems:
            error_message = f"Se encontraron los siguientes problemas: {'/n'.join(problems)}"
            logger.error(error_message)
            raise TypeError(error_message)

        logger.info("Tipos de datos validados correctamente!")

        return df