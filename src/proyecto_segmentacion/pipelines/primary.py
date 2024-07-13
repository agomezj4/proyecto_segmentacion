from typing import Dict, Any

import pandas as pd
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PipelinePrimary:

    # 1. Recategorizar columnas
    @staticmethod
    def recategorize_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Recategoriza los valores de las columnas especificadas en un DataFrame basado en
        las categorías proporcionadas en los parámetros.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada del cual se recategorizarán los valores.
        params : Dict[str, Any]
            Parámetros primary.

        Returns
        -------
        pd.DataFrame: DataFrame con los valores recategorizados.
        """
        logger.info("Iniciando el proceso de recategorización...")

        # Parámetros
        campos_recategorizacion = params['campos_recategorizacion']

        for campo, recategorizaciones in campos_recategorizacion.items():
            if campo in df.columns:
                logger.info(f"Recategorizando el campo '{campo}'")
                for recategorizacion in recategorizaciones:
                    for original, nuevo in recategorizacion.items():
                        df[campo] = df[campo].replace(original, nuevo)
                        logger.info(f"Recategorizado valor '{original}' a '{nuevo}' en el campo '{campo}'")
            else:
                logger.warning(f"El campo '{campo}' no se encuentra en el DataFrame y será omitido.")

        logger.info("Recategorización completada!")
        return df

    #2. Impurtar datos faltantes
    @staticmethod
    def impute_missing_values_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Imputa los valores faltantes en un DataFrame de Pandas basado en el tipo de cada columna
        y excluye el campo de ID especificado en los parámetros.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de Pandas sobre el cual se realizarán las imputaciones.
        params : Dict[str, Any]
            Parámetros primary.

        Returns
        -------
        pd.DataFrame
            DataFrame con los valores faltantes imputados.
        """
        logger.info("Iniciando el proceso de imputación de valores faltantes...")

        # Obtener el nombre del campo de ID desde los parámetros
        id_field = params['id_cliente'][0]

        # Identificar columnas con valores nulos, excluyendo el campo de ID
        columns_with_nulls = [
            col for col in df.columns
            if col != id_field and (
                    df[col].isnull().any() or
                    (df[col] == '').any() or
                    (df[col].astype(str).str.lower() == 'null').any() or
                    (df[col].astype(str).str.lower() == 'none').any()
            )
        ]

        logger.info(f"Columnas con valores nulos identificadas: {columns_with_nulls}")

        # Procesar cada columna con valores nulos
        for col in columns_with_nulls:
            col_type = df[col].dtype

            if pd.api.types.is_string_dtype(col_type):
                # Imputar con la moda para columnas de tipo string
                mode_value = df[col].mode().iloc[0]  # Obtener la moda
                df[col] = df[col].replace(['', 'NULL', 'Null', 'None', None], mode_value)
                df[col] = df[col].fillna(mode_value)
                logger.info(f"Imputados valores nulos en la columna '{col}' con la moda: {mode_value}")

            elif pd.api.types.is_integer_dtype(col_type):
                # Imputar con la mediana para columnas de tipo int64
                median_value = df[col].median()
                median_value = round(median_value)  # Redondear al entero más cercano si es necesario
                df[col] = df[col].replace(['', 'NULL', 'Null', 'None', None], median_value)
                df[col] = df[col].fillna(median_value)
                logger.info(f"Imputados valores nulos en la columna '{col}' con la mediana redondeada: {median_value}")

            elif pd.api.types.is_float_dtype(col_type):
                # Imputar con la mediana para columnas de tipo float64
                median_value = df[col].median()
                df[col] = df[col].replace(['', 'NULL', 'Null', 'None', None], median_value)
                df[col] = df[col].fillna(median_value)
                logger.info(f"Imputados valores nulos en la columna '{col}' con la mediana: {median_value}")

        # Verificar y manejar nulos en el campo de ID
        if id_field and df[id_field].isnull().any().any():
            logger.warning(f"Se encontraron valores nulos en el campo de ID '{id_field}'.")

            # Recomendación técnica: eliminar registros con ID nulo
            df_clean = df.dropna(subset=[id_field])
        else:
            df_clean = df

        logger.info("Imputación de valores faltantes completada!")
        return df_clean

    # 3. Eliminar clientes con id duplicado
    @staticmethod
    def remove_duplicates_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Elimina duplicados en el DataFrame basado en la combinación de los campos 'id_cliente' y 'id_sesion',
        quedándose con el último registro para cada combinación duplicada.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada del cual se removerán los duplicados.
        params : Dict[str, Any]
            Parámetros primary que incluyen 'id_cliente' e 'id_sesion'.

        Returns
        -------
        pd.DataFrame: DataFrame sin duplicados en la combinación de 'id_cliente' e 'id_sesion'.
        """
        logger.info("Iniciando el proceso de eliminación de duplicados...")

        # Obtener los nombres de los campos de ID desde los parámetros
        id_cliente = params['id_cliente'][0]
        id_sesion = params['id_sesion'][0]

        # Verificar si hay duplicados en la combinación de los campos de ID
        has_duplicates = df.duplicated(subset=[id_cliente, id_sesion]).any()

        if has_duplicates:
            logger.info(
                f"Se encontraron duplicados en la combinación de los campos de ID '{id_cliente}' y '{id_sesion}'. "
                f"Eliminando duplicados y conservando el último registro...")

            # Eliminar duplicados y conservar el último registro
            df_clean = df.drop_duplicates(subset=[id_cliente, id_sesion], keep='last')
            logger.info(f"Duplicados eliminados. Total de registros después de la eliminación: {len(df_clean)}")
        else:
            logger.info(f"No se encontraron duplicados en la combinación de los campos '{id_cliente}' y '{id_sesion}'.")
            df_clean = df

        logger.info("Finaliza la verificación de duplicados!")
        return df_clean

    # 4. Remover outliers
    @staticmethod
    def remove_outliers_pd(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remueve outliers de un DataFrame en múltiples columnas numéricas (float64 e int64)
        utilizando el rango intercuartílico (IQR).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada del cual se removerán los outliers.

        Returns
        -------
        pd.DataFrame: DataFrame filtrado sin outliers en las columnas especificadas.
        """
        logger.info("Iniciando el proceso de eliminación de outliers...")

        # Filtrar solo las columnas numéricas (float64 e int64)
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        logger.info(f"Columnas numéricas identificadas para la eliminación de outliers: {numeric_columns.tolist()}")

        df_clean = df.copy()  # Hacemos una copia del DataFrame original para no modificarlo directamente

        for column_name in numeric_columns:
            # Calcular el primer y tercer cuartil (Q1 y Q3)
            q1 = df_clean[column_name].quantile(0.25)
            q3 = df_clean[column_name].quantile(0.75)
            iqr = q3 - q1

            # Determinar los límites inferior y superior para identificar outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Verificar si existen outliers en la columna
            has_outliers = df_clean[
                (df_clean[column_name] < lower_bound) | (df_clean[column_name] > upper_bound)].any().any()

            if has_outliers:
                logger.info(f"Se encontraron outliers en la columna '{column_name}'. Removiendo outliers...")
                # Filtrar el DataFrame para remover outliers
                df_clean = df_clean[(df_clean[column_name] >= lower_bound) & (df_clean[column_name] <= upper_bound)]
            else:
                logger.info(f"No se encontraron outliers en la columna '{column_name}'.")

        logger.info("Eliminación de outliers completada.")
        return df_clean

