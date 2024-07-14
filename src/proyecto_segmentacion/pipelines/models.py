from typing import Dict, Any, Tuple
import pandas as pd
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PipelineModels:

    # 1. K-Means Clustering
    @staticmethod
    def train_kmeans_pd(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[KMeans, float]:
        """
        Entrena un modelo K-Means y determina el número óptimo de clusters utilizando el método del codo.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada para el entrenamiento del modelo K-Means.
        params : Dict[str, Any]
            Diccionario de parámetros models.

        Returns
        -------
        Tuple[KMeans, float]
            Modelo K-Means entrenado y el puntaje de coeficiente de silueta.
        """
        logger.info("Iniciando el proceso de entrenamiento K-Means...")

        # Parámetros
        k_means_params = params['K_Means']
        exclude_columns = params['no_columns']

        # Excluir columnas no relevantes
        df = df.drop(columns=exclude_columns)

        # Verificar que el rango de k sea adecuado
        k_range = list(range(k_means_params['k_range'][0], k_means_params['k_range'][1] + 1))
        logger.info(f"Rango de k generado: {k_range}")
        if len(k_range) < 3:
            logger.error("El rango de valores de k es demasiado pequeño. Debe haber al menos 3 valores.")
            raise ValueError("El rango de valores de k es demasiado pequeño. Debe haber al menos 3 valores.")

        # Encontrar el número óptimo de clusters con el método del codo
        distortions = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=k_means_params['random_state'])
            kmeans.fit(df)
            distortions.append(kmeans.inertia_)

        # Verificar que haya suficientes elementos en distortions
        if len(distortions) < 3:
            logger.error("No hay suficientes elementos en distortions para encontrar el codo.")
            raise ValueError("No hay suficientes elementos en distortions para encontrar el codo.")

        # Determinar el codo en la gráfica de distorsión
        optimal_k = PipelineModels.find_elbow(distortions)
        logger.info(f"Número óptimo de clusters determinado: {optimal_k}")

        # Entrenar el modelo K-Means con el número óptimo de clusters
        kmeans = KMeans(
            n_clusters=optimal_k,
            init=k_means_params['init'],
            n_init=k_means_params['n_init'],
            max_iter=k_means_params['max_iter'],
            tol=k_means_params['tol'],
            random_state=k_means_params['random_state'],
            algorithm=k_means_params['algorithm']
        )
        kmeans.fit(df)

        # Obtener las métricas de coeficiente de silueta
        labels = kmeans.labels_
        sil_score = silhouette_score(df, labels)
        logger.info(f"Coeficiente de silueta obtenido: {sil_score}")

        return kmeans, sil_score

    @staticmethod
    def find_elbow(distortions: list) -> int:
        """
        Encuentra el codo en la gráfica de distorsiones.

        Parameters
        ----------
        distortions : list
            Lista de distorsiones para diferentes números de clusters.

        Returns
        -------
        int
            Número óptimo de clusters determinado por el método del codo.
        """
        logger.info("Encontrando el codo en la gráfica de distorsiones...")

        # Diferencias en las distorsiones
        diffs = np.diff(distortions)

        # Verificar que haya suficientes elementos en diffs
        if len(diffs) < 2:
            logger.error("No hay suficientes elementos en diffs para encontrar el codo.")
            raise ValueError("No hay suficientes elementos en diffs para encontrar el codo.")

        # Doble diferencia (aceleración) para encontrar el punto donde la reducción se vuelve menos pronunciada
        d2_diffs = np.diff(diffs)

        # El índice del codo es el punto donde la aceleración cambia significativamente
        if len(d2_diffs) == 0:
            logger.error("No hay suficientes elementos en d2_diffs para encontrar el codo.")
            raise ValueError("No hay suficientes elementos en d2_diffs para encontrar el codo.")

        optimal_k = np.argmin(d2_diffs) + 2  # +2 porque np.diff reduce el tamaño del array en 1 cada vez

        logger.info(f"Codo encontrado en el número de clusters: {optimal_k}")
        return optimal_k

    # 2. Gaussian Mixture Model
    @staticmethod
    def train_gmm_pd(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[GaussianMixture, float]:
        """
        Entrena un modelo de Gaussian Mixture Model (GMM) y determina el número óptimo de componentes
        utilizando el Criterio de Información Bayesiano (BIC).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada para el entrenamiento del modelo GMM.
        params : Dict[str, Any]
            Diccionario de parámetros models.

        Returns
        -------
        Tuple[GaussianMixture, float]
            Modelo GMM entrenado y el puntaje de coeficiente de silueta.
        """
        logger.info("Iniciando el proceso de entrenamiento GMM...")

        # Parámetros
        gmm_params = params['Gaussian_Mixture_Model']
        exclude_columns = params['no_columns']

        # Excluir columnas no relevantes
        df = df.drop(columns=exclude_columns)
        logger.info("Columnas excluidas: %s", exclude_columns)

        # Encontrar el número óptimo de componentes con el Criterio de Información Bayesiano (BIC)
        optimal_k = PipelineModels.find_optimal_gmm_components(df, gmm_params)
        logger.info(f"Número óptimo de componentes determinado: {optimal_k}")

        # Entrenar el modelo GMM con el número óptimo de componentes
        gmm = PipelineModels.configure_and_train_gmm(df, gmm_params, optimal_k)

        # Obtener las métricas de coeficiente de silueta
        labels = gmm.predict(df)
        sil_score = silhouette_score(df, labels)
        logger.info(f"Coeficiente de silueta obtenido: {sil_score}")

        return gmm, sil_score

    @staticmethod
    def find_optimal_gmm_components(df: pd.DataFrame, params: Dict[str, Any]) -> int:
        """
        Encuentra el número óptimo de componentes para el modelo GMM utilizando el Criterio de Información Bayesiano (BIC).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada.
        params : Dict[str, Any]
            Diccionario de parámetros para el modelo GMM.

        Returns
        -------
        int
            Número óptimo de componentes determinado por el menor BIC.
        """
        logger.info("Calculando el número óptimo de componentes utilizando BIC...")

        bics = []
        x = range(1, params['max_components'] + 1)
        for k in x:
            gmm = PipelineModels.configure_and_train_gmm(df, params, k)
            bics.append(gmm.bic(df))
            logger.debug(f"BIC para {k} componentes: {bics[-1]}")

        # Seleccionar el número óptimo de componentes (menor BIC)
        optimal_k = np.argmin(bics) + 1
        logger.info(f"Menor BIC encontrado: {bics[optimal_k - 1]} para {optimal_k} componentes")

        return optimal_k

    @staticmethod
    def configure_and_train_gmm(df: pd.DataFrame, params: Dict[str, Any], n_components: int) -> GaussianMixture:
        """
        Configura y entrena un modelo GMM con el número de componentes especificado.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada.
        params : Dict[str, Any]
            Diccionario de parámetros para el modelo GMM.
        n_components : int
            Número de componentes para el modelo GMM.

        Returns
        -------
        GaussianMixture
            Modelo GMM entrenado.
        """
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=params['covariance_type'],
            tol=params['tol'],
            reg_covar=params['reg_covar'],
            max_iter=params['max_iter'],
            n_init=params['n_init'],
            init_params=params['init_params'],
            warm_start=params['warm_start'],
            verbose=params['verbose'],
            verbose_interval=params['verbose_interval'],
            random_state=params['random_state']
        )
        gmm.fit(df)
        return gmm

    # 3. DBSCAN
    @staticmethod
    def train_dbscan_pd(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[DBSCAN, float]:
        """
        Entrena un modelo DBSCAN y calcula el coeficiente de silueta para los datos no considerados como ruido.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada para el entrenamiento del modelo DBSCAN.
        params : Dict[str, Any]
            Diccionario de parámetros para el modelo DBSCAN y exclusión de columnas.

        Returns
        -------
        Tuple[DBSCAN, float]
            Modelo DBSCAN entrenado y el puntaje de coeficiente de silueta.
        """
        logger.info("Iniciando el proceso de entrenamiento DBSCAN...")

        # Parámetros
        dbscan_params = params['DBSCAN']
        exclude_columns = params['no_columns']

        # Excluir columnas no relevantes
        df = df.drop(columns=exclude_columns)

        # Aplicar DBSCAN
        dbscan = DBSCAN(
            eps=dbscan_params['eps'],
            min_samples=dbscan_params['min_samples'],
            metric=dbscan_params['metric'],
            algorithm=dbscan_params['algorithm'],
            leaf_size=dbscan_params['leaf_size']
        )
        labels = dbscan.fit_predict(df)

        # Filtrar los puntos etiquetados como ruido para el cálculo del índice de silueta
        mask = labels != -1
        filtered_data = df[mask]
        filtered_labels = labels[mask]

        # Calcular métricas
        if len(set(filtered_labels)) > 1:  # Silhouette score no se puede calcular con un solo cluster
            sil_score = silhouette_score(filtered_data, filtered_labels)
            logger.info(f"Coeficiente de silueta obtenido: {sil_score}")
        else:
            sil_score = -1  # Valor indicativo de que no se pudo calcular el silhouette score
            logger.warning("No se pudo calcular el coeficiente de silueta porque hay un solo cluster.")

        return dbscan, sil_score

