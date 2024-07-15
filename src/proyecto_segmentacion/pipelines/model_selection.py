from typing import List, Tuple, Any, Dict

import numpy as np
import logging
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from bayes_opt import BayesianOptimization

from src.proyecto_segmentacion.utils import Utils

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PipelineModelSelection:

    # 1. Compara los algoritmos en caso que la prueba ANOVA no sea significativa
    @staticmethod
    def compare_models(artifacts: List[Tuple[Any, List[float], pd.DataFrame]]) -> Tuple[Any, float, pd.DataFrame]:
        """
        Compara los modelos basándose en el coeficiente de silueta promedio y selecciona el mejor.

        Parameters
        ----------
        artifacts: List[Tuple[Any, List[float], pd.DataFrame]]
            Lista de tuplas que contienen el modelo entrenado, lista de coeficientes de silueta y el
            DataFrame con las predicciones correspondientes.

        Returns
        -------
        Tuple[Any, float, pd.DataFrame]
            El modelo seleccionado, coeficiente de silueta promedio y el DataFrame correspondiente.
        """
        if not artifacts or len(artifacts) != 3:
            logger.error("Se deben proporcionar exactamente tres artefactos de modelos para comparar.")
            raise ValueError("Se deben proporcionar exactamente tres artefactos de modelos para comparar.")

        # Calcular el promedio de los coeficientes de silueta para cada modelo
        artifacts_avg = [(artifact[0], np.mean(artifact[1]), artifact[2]) for artifact in artifacts]

        # Ordenar los artefactos por el coeficiente de silueta promedio
        artifacts_avg.sort(key=lambda x: x[1], reverse=True)
        best_model, best_sil_score, best_df = artifacts_avg[0]
        logger.info(f"Mejor modelo seleccionado con coeficiente de silueta promedio: {best_sil_score}")

        return best_model, best_sil_score, best_df

    # 2. Realiza la prueba ANOVA para comparar los coeficientes de silueta entre los modelos
    @staticmethod
    def anova_test(sil_scores: List[List[float]], params: Dict[str, Any]) -> Tuple[float, bool, List[float]]:
        """
        Realiza una prueba ANOVA de una vía para comparar los coeficientes de silueta entre los modelos.

        Parameters
        ----------
        sil_scores : List[List[float]]
            Lista de listas de coeficientes de silueta para los modelos.
        params : Dict[str, Any]
            Diccionario de parámetros model selection.

        Returns
        -------
        Tuple[float, bool, List[float]]
            El valor p resultante de la prueba ANOVA, un booleano indicando si hay diferencias significativas,
            y una lista con los promedios de silueta de cada modelo.
        """

        # Parámetros
        threshold = params['threshold_significance']

        if len(sil_scores) != 3:
            logger.error("Se necesitan exactamente tres listas de coeficientes de silueta para realizar la prueba ANOVA.")
            raise ValueError("Se necesitan exactamente tres listas de coeficientes de silueta para realizar la prueba ANOVA.")

        # Realizar la prueba ANOVA
        f_val, p_val = f_oneway(*sil_scores)
        logger.info(f"Resultado de la prueba ANOVA: f_val={f_val}, p_val={p_val}")

        # Calcular los promedios de los coeficientes de silueta para cada modelo
        sil_means = [np.mean(scores) for scores in sil_scores]

        return p_val, p_val < threshold, sil_means

    # 3. Realiza la prueba post-hoc de Tukey para identificar qué grupo tiene la mayor diferencia significativa
    @staticmethod
    def tukey_test(sil_scores: List[List[float]]) -> int:
        """
        Realiza una prueba post-hoc de Tukey para identificar qué grupo tiene la mayor diferencia significativa.

        Parameters
        ----------
        sil_scores : List[List[float]]
            Lista de listas de coeficientes de silueta para los modelos.

        Returns
        -------
        int
            Índice del modelo con la mayor diferencia significativa.
        """
        all_scores = []
        groups = []
        for i, scores in enumerate(sil_scores):
            all_scores.extend(scores)
            groups.extend([i] * len(scores))

        tukey_result = tukey_hsd(all_scores, groups)
        logger.info(f"Resultado de la prueba de Tukey: {tukey_result}")

        # Encontrar el índice del grupo con la mayor media significativa
        means = [np.mean(scores) for scores in sil_scores]
        best_index = np.argmax(means)

        return best_index

    # 4. Selecciona el mejor modelo basándose en el coeficiente de silueta o un resultado de prueba estadística
    @staticmethod
    def select_best_model_pd(artifacts: List[Tuple[Any, List[float], pd.DataFrame]], params: Dict[str, Any]
                             ) -> Tuple[Any, float, pd.DataFrame]:
        """
        Selecciona el mejor modelo basándose en el coeficiente de silueta y un resultado de prueba estadística.

        Parameters
        ----------
        artifacts : List[Tuple[Any, List[float], pd.DataFrame]]
            Lista de tuplas que contienen el modelo entrenado, su lista de coeficientes de silueta y el
            DataFrame correspondiente.
        params : Dict[str, Any]
            Diccionario de parámetros model selection.

        Returns
        -------
        Tuple[Any, float, pd.DataFrame]
            El modelo seleccionado, su coeficiente de silueta promedio y el DataFrame correspondiente.
        """

        logger.info("Seleccionando el mejor modelo...")

        if len(artifacts) != 3:
            logger.error("Se deben proporcionar exactamente tres artefactos de modelos para comparar.")
            raise ValueError("Se deben proporcionar exactamente tres artefactos de modelos para comparar.")

        # Obtener las listas de coeficientes de silueta
        sil_scores = [artifact[1] for artifact in artifacts]

        # Realizar prueba ANOVA
        p_val, has_significant_difference, sil_means = PipelineModelSelection.anova_test(sil_scores, params)

        if has_significant_difference:
            logger.info("Diferencia significativa encontrada entre los coeficientes de silueta de los modelos.")
            best_index = PipelineModelSelection.tukey_test(sil_scores)
            best_model, best_sil_score_list, best_df = artifacts[best_index]
            best_sil_score = np.mean(best_sil_score_list)
            logger.info(f"El modelo con diferencia significativa es el modelo {best_model} "
                        f"con un coeficiente de silueta promedio de {best_sil_score}")

        else:
            logger.info("No se encontró diferencia significativa entre los coeficientes de silueta de los modelos.")
            # Si no hay diferencia significativa, simplemente selecciona el modelo con el mejor silueta promedio
            best_model, best_sil_score, best_df = PipelineModelSelection.compare_models(artifacts)

        logger.info("Selección de mejor modelo completada!")

        return best_model, best_sil_score, best_df

    # 5. Optimiza los hiperparámetros del mejor modelo seleccionado
    @staticmethod
    def optimize_train_best_gmm_pd(
            df: pd.DataFrame,
            params: Dict[str, Any]
    ) -> Tuple[Any, float, pd.DataFrame]:
        """
        Optimiza y entrena un modelo Gaussian Mixture Model (GMM) usando optimización bayesiana para
        encontrar los mejores hiperparámetros basados en el coeficiente de silueta.
        Evalúa el modelo en el conjunto de datos proporcionado.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada para el entrenamiento del modelo GMM.
        params : Dict[str, Any]
            Diccionario de parámetros para el modelo GMM y exclusión de columnas.

        Returns
        -------
        Tuple[Any, float, pd.DataFrame]
            Mejor modelo GMM entrenado, promedio de coeficientes de silueta y dataframe con los clusters asignados.
        """

        logger.info("Iniciando la optimización y el entrenamiento GMM...")

        # Parámetros
        gmm_params = params['Gaussian_Mixture_Model']
        exclude_columns = params['no_columns']
        seed = params["random_state"]
        exploration_space = gmm_params["exploration_space"]
        init_points = gmm_params["init_points"]

        # Preparar los datos
        df, cliente_id = Utils.prepare_data(df, exclude_columns)

        def calculate_silhouette_score(data, cluster_labels):
            mask = cluster_labels != -1
            filtered_data = data[mask]
            filtered_labels = cluster_labels[mask]

            if len(set(filtered_labels)) > 1:
                return silhouette_score(filtered_data, filtered_labels)
            else:
                return -1

        covariance_types = exploration_space['covariance_type']
        exploration_space_mapped = {
            'n_components': tuple(exploration_space['n_components']),
            'covariance_type': (0, len(covariance_types) - 1),
            'tol': tuple(exploration_space['tol'])
        }

        def gmm_evaluate(n_components, covariance_type, tol):
            cov_type = covariance_types[int(covariance_type)]
            gmm = GaussianMixture(
                n_components=int(n_components),
                covariance_type=cov_type,
                tol=tol,
                random_state=seed
            )
            cluster_labels = gmm.fit_predict(df)
            sil_score = calculate_silhouette_score(df, cluster_labels)
            return sil_score

        optimizer = BayesianOptimization(
            f=gmm_evaluate,
            pbounds=exploration_space_mapped,
            random_state=seed,
            verbose=2
        )
        optimizer.maximize(n_iter=gmm_params['number_of_iterations'], init_points=init_points)

        best_params = optimizer.max["params"]
        best_score = optimizer.max["target"]

        logger.info(f"Mejores parámetros GMM: {best_params} con un coeficiente de silueta de {best_score}")

        best_gmm = GaussianMixture(
            n_components=int(best_params["n_components"]),
            covariance_type=covariance_types[int(best_params["covariance_type"])],
            tol=best_params["tol"],
            random_state=seed
        )
        final_labels = best_gmm.fit_predict(df)
        avg_sil_score = calculate_silhouette_score(df, final_labels)

        clusters_df = Utils.join_clusters(cliente_id, final_labels)

        return best_gmm, avg_sil_score, clusters_df

