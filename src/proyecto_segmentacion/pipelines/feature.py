from typing import Dict, Any

import pandas as pd
import re
import logging
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PipelineFeature:

    # 1. Creación nuevas características
    @staticmethod
    def features_new_pd(
            df: pd.DataFrame,
            params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Calcula nuevas características para cada cliente

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame de pandas que contiene las características del cliente
        params: Dict[str, Any]
            Diccionario de parámetros feature

        Returns
        -------
        pd.DataFrame: DataFrame con las nuevas columnas agregadas.
        """
        logger.info("Iniciando el cálculo de nuevas características ..")

        # 1. Rango de horas: categoriza la hora de visita en madrugada, mañana, tarde, noche.
        df['hora_rango'] = pd.cut(df[params['data_features']['hora_visita']],
                                  bins=params['data_features']['bins_hora'],
                                  labels=params['data_features']['labels_hora'],
                                  include_lowest=True).astype('object')
        logger.info("Rango de horas calculado.")

        # 2. Interacciones por página: promedio de interacciones por página vista.
        df['interacciones_por_pagina'] = df[params['data_features']['interacciones_sesion']] / df[
            params['data_features']['pag_vistas_sesion']]
        logger.info("Interacciones por página calculado.")

        # 3. Fuente de tráfico: combina el canal de agrupación y el medio de tráfico para una
        # visión más detallada de la fuente de tráfico.
        df['fuente_trafico'] = df[params['data_features']['canal_agrupacion']] + ' - ' + df[
            params['data_features']['medio_trafico']]
        logger.info("Fuente de tráfico calculado.")

        # 4. Visitas en fines de semana: marca binaria para saber si el cliente visita en fines de semana.
        df['clt_visita_fds'] = df[params['data_features']['tasa_visitas_fds']].apply(lambda x: 1 if x > 0.0 else 0)
        logger.info("Marca visitas en fines de semana calculado.")

        # 5. Rebote: marca binaria para saber si el cliente ha rebotado en la página.
        df['clt_rebota'] = df[params['data_features']['tasa_rebote']].apply(lambda x: 1 if x > 0.0 else 0)
        logger.info("Marca rebote calculado.")

        # 6. Modificación de dispositivo_movil: cambiar False por 0 y True por 1, y cambiar tipo a int64.
        df['dispositivo_movil'] = df['dispositivo_movil'].astype(int)
        logger.info("Modificación de dispositivo_movil calculada.")

        logger.info("Finalizado el cálculo de nuevas características.")
        return df

    # 2. Encoding de variables categóricas

    # 2.1. Categorización acumulativa
    @staticmethod
    def cumulatively_categorise_pd(column: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """
        Categoriza acumulativamente una columna de un DataFrame de Pandas, reemplazando los valores
        que no cumplen con el umbral especificado.

        Parameters
        ----------
        column : pd.Series
            Columna de un DataFrame de Pandas que se categorizará acumulativamente
        params: Dict[str, Any]
            Diccionario de parámetros featuring

        Returns
        -------
        pd.Series
            Columna de un DataFrame de Pandas con la categorización acumulativa aplicada.
        """
        logger.info(f"Empieza el proceso de categorización acumulativa para el campo '{column.name}'...")

        # Parámetros
        threshold = params['threshold']
        replacement_value = params['value']

        # Calculamos el valor de umbral basado en el porcentaje dado
        threshold_value = int(threshold * len(column))

        # Calculamos los conteos y ordenamos de forma descendente
        counts = column.value_counts().sort_values(ascending=False)

        # Acumulamos las frecuencias hasta llegar o superar el umbral
        cumulative_counts = counts.cumsum()
        valid_categories = cumulative_counts[cumulative_counts <= threshold_value].index.tolist()

        # Creamos una lista con las categorías válidas más el valor de reemplazo
        valid_categories.append(replacement_value)

        # Reemplazamos los valores que no están en la lista de categorías válidas
        new_column = column.apply(lambda x: x if x in valid_categories else replacement_value)

        logger.info(f"Categorización acumulativa completada para el campo '{column.name}'!")

        return new_column

    # 2.2. One Hot Encoding
    @staticmethod
    def replace_spaces_with_underscores(category):
        return re.sub(r'\s+', '_', category)

    @staticmethod
    def one_hot_encoding_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Aplica One Hot Encoding a las columnas especificadas en el diccionario de parámetros.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de Pandas al que se le aplicará One Hot Encoding
        params: Dict[str, Any]
            Diccionario de parámetros featuring

        Returns
        -------
        pd.DataFrame: DataFrame con las columnas transformadas.
        """
        logger.info("Iniciando One Hot Encoding...")

        # Parámetros
        cum_cat = params['cum_cat']
        no_cum_cat = cum_cat['no_cum_cat']
        one_hot_encoder_columns = [nombre for nombre in df.columns if df[nombre].dtype == 'object']

        # Filtramos las columnas a excluir
        one_hot_encoder_columns = [col for col in one_hot_encoder_columns if col not in no_cum_cat]

        for var in one_hot_encoder_columns:
            if var not in df.columns:
                logger.error(f"La columna '{var}' no existe en el DataFrame.")
                raise KeyError(f"La columna '{var}' no existe en el DataFrame.")

            # `cumulatively_categorise` es una función definida que categoriza y luego transforma en códigos enteros.
            df[var] = PipelineFeature.cumulatively_categorise_pd(df[var], cum_cat)

        # Inicializamos el OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, drop=None)
        encoded_columns = encoder.fit_transform(df[one_hot_encoder_columns])
        encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(one_hot_encoder_columns))

        # Eliminamos las columnas originales y unimos las nuevas columnas codificadas
        df = df.drop(columns=one_hot_encoder_columns).reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)

        logger.info("One Hot Encoding completado!")

        return df

    # 3. Escalado de variables numéricos
    @staticmethod
    def min_max_scaler_pd(df: pd.DataFrame) -> pd.DataFrame:
        """
        Estandariza las columnas numéricas (excluyendo binarias) de un DataFrame utilizando el método Min-Max Scaler.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de Pandas que se estandarizará.

        Returns
        -------
        pd.DataFrame
            DataFrame estandarizado.
        """
        logger.info("Iniciando la estandarización con Min-Max Scaler...")

        # Identificar las columnas numéricas
        numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns

        # Filtrar solo las columnas numéricas no binarias (excluyendo aquellas que solo toman valores 0 y 1)
        numeric_cols = [col for col in numeric_cols if
                        not ((df[col].nunique() == 2) & (df[col].isin([0, 1]).sum() == len(df)))]

        # Crear una copia del DataFrame para evitar el SettingWithCopyWarning
        df_copy = df.copy()

        # Aplicar Min-Max Scaler solo a las columnas numéricas no binarias
        for col in numeric_cols:
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            range_val = max_val - min_val
            if range_val != 0:  # Evita la división por cero en caso de que todas las entradas en una columna sean iguales
                df_copy[col] = df_copy[col].astype('float64')  # Convertir la columna a float64
                df_copy.loc[:, col] = (df_copy[col] - min_val) / range_val

        logger.info("Estandarización con Min-Max Scaler completada!")

        return df_copy
