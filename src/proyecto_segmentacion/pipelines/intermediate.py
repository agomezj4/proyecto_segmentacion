from typing import Dict, Any

import pandas as pd
import unicodedata
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PipelineIntermediate:

    # 1. Re- nombrar
    @staticmethod
    def change_names_pd(df: pd.DataFrame, tag_dict: pd.DataFrame) -> pd.DataFrame:
        """
        Cambia los nombres de las columnas de un DataFrame según un tag dictionary y devuelve
        un nuevo DataFrame con los nombres cambiados utilizando Pandas.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de Pandas del cual se cambiarán los nombres de las columnas
        tag_dict: pd.DataFrame
            Diccionario de etiquetas

        Returns
        -------
        pd.DataFrame: Un nuevo DataFrame con las columnas renombradas según el tag dictionary
        """
        logger.info("Iniciando el cambio de nombres...")

        # Filtra las filas del tag dictionary donde "source" sea "raw" y selecciona las columnas "tag" y "name"
        tag_dict_filtered = tag_dict[tag_dict["source"] == "raw"][["tag", "name"]]

        # Crea un diccionario de mapeo para cambiar los nombres de las columnas
        col_mapping = dict(zip(tag_dict_filtered['tag'], tag_dict_filtered['name']))

        # Cambia los nombres de las columnas según el tag dictionary "raw"
        data = df.rename(columns=col_mapping)

        logger.info("Nombres de columnas cambiados!")

        return data

    # 2. Cambiar tipos de datos
    @staticmethod
    def change_dtype_pd(df: pd.DataFrame, tag_dict: pd.DataFrame) -> pd.DataFrame:
        """
        Cambia el tipo de datos de cada columna en un DataFrame al tipo de datos especificado
        en el tag dictionary usando Pandas.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de Pandas del cual se cambiarán los tipos de datos
        tag_dict: pd.DataFrame
            Diccionario de etiquetas

        Returns
        -------
        pd.DataFrame: DataFrame con las columnas cambiadas al nuevo tipo de datos.
        """
        logger.info("Iniciando el cambio de tipos de datos...")

        # Filtrar tag_dict para incluir solo las filas con "source" igual a "raw"
        tag_dict = tag_dict[tag_dict["source"] == "raw"]

        # Crear un diccionario de mapeo de tipos de datos
        type_mapping = dict(zip(tag_dict['name'], tag_dict['data_type_new']))

        # Cambiar el tipo de datos de cada columna según el mapeo
        for col in df.columns:
            if col in type_mapping:
                try:
                    new_type = type_mapping[col]
                    if new_type == 'object':
                        df[col] = df[col].astype(str)
                    elif new_type == 'bool':
                        df[col] = df[col].astype(bool)
                    elif new_type == 'float64':
                        df[col] = df[col].astype(float)
                    elif new_type == 'int64':
                        df[col] = df[col].astype(int)
                    elif new_type == 'uint64':
                        df[col] = df[col].astype('uint64')
                    else:
                        df[col] = df[col].astype(new_type)
                    logger.info(f"Columna {col} cambiada a {new_type}")
                except Exception as e:
                    logger.error(f"Error al intentar castear la columna {col} a {new_type}: {e}")

        logger.info("Dtypes cambiados!")

        return df

    # 3. Eliminar acentos
    @staticmethod
    def delete_accents_pd(df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina los acentos de las columnas identificadas como "str" de un DataFrame de Pandas.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de Pandas del cual se eliminarán los acentos

        Returns
        -------
        pd.DataFrame
            DataFrame de Pandas con los acentos eliminados de las columnas especificadas.
        """
        logger.info("Iniciando la eliminación de acentos...")

        # Función para eliminar acentos
        def remove_accents(input_str):
            nfkd_form = unicodedata.normalize('NFKD', input_str)
            return "".join([char for char in nfkd_form if not unicodedata.combining(char)])

        # Aplicar la eliminación de acentos solo a las columnas de tipo string
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(lambda x: remove_accents(x) if isinstance(x, str) else x)

        logger.info("Acentos eliminados!")

        return df

