from typing import Dict, Any

import pandas as pd
import logging

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
                                  include_lowest=True)
        logger.info("Rango de horas calculado.")

        # 2. Interacciones por página: promedio de interacciones por página vista.
        df['interacciones_por_pagina'] = df[params['data_features']['interacciones_sesion']] / df[
            params['data_features']['pag_vistas_sesion']]
        logger.info("Interacciones por página calculado.")

        # 3. Fuente de tráfico: combina el canal de agrupación y el medio de tráfico para una
        # vision más detallada de la fuente de tráfico.
        df['fuente_trafico'] = df[params['data_features']['canal_agrupacion']] + ' - ' + df[
            params['data_features']['medio_trafico']]
        logger.info("Fuente de tráfico calculado.")

        # 4. Visitas en fines de semana: marca binaria para saber si el cliente visita en fines de semana.
        df['clt_visita_fds'] = df[params['data_features']['tasa_visitas_fds']].apply(lambda x: 1 if x > 0.0 else 0)
        logger.info("Marca visitas en fines de semana calculado.")

        # 5. Rebote: marca binaria para saber si el cliente ha rebotado en la página.
        df['clt_rebota'] = df[params['data_features']['tasa_rebote']].apply(lambda x: 1 if x > 0.0 else 0)
        logger.info("Marca rebote calculado.")

        logger.info("Finalizado el cálculo de nuevas características.")
        return df

    # 2. Escalado de variables numéricos
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
