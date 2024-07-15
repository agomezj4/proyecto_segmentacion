import os
from .utils import Utils

logger = Utils.setup_logging()
Utils.add_src_to_path()
project_root = Utils.get_project_root()

parameters_directory = os.path.join(project_root, 'src', 'parameters')
data_raw_directory = os.path.join(project_root, 'data', '01_raw')
data_intermediate_directory = os.path.join(project_root, 'data', '02_intermediate')
data_primary_directory = os.path.join(project_root, 'data', '03_primary')
data_feature_directory = os.path.join(project_root, 'data', '04_feature')
data_models_directory = os.path.join(project_root, 'data', '05_models')
data_model_selection_directory = os.path.join(project_root, 'data', '06_model_selection')

parameters = Utils.load_parameters(parameters_directory)


class PipelineOrchestration:

    # 1. Pipeline Raw
    @staticmethod
    def run_pipeline_raw():
        from .pipelines.raw import PipelineRaw

        tag_dict_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['tag_dict_path'])
        customers_data_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['data_customers_path'])

        tag_dict = Utils.load_data(tag_dict_path)
        data_customers = Utils.load_data(customers_data_path)

        data_validate_tags = PipelineRaw.validate_tags_pd(data_customers, tag_dict)
        data_raw = PipelineRaw.validate_dtypes_pd(data_validate_tags, tag_dict)

        raw_data_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['raw_data_path'])
        Utils.save_data(data_raw, raw_data_path)

    # 2. Pipeline Intermediate
    @staticmethod
    def run_pipeline_intermediate():
        from .pipelines.intermediate import PipelineIntermediate

        raw_data_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['raw_data_path'])
        tag_dict_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['tag_dict_path'])

        data_raw = Utils.load_data(raw_data_path)
        tag_dict = Utils.load_data(tag_dict_path)

        data_change_names = PipelineIntermediate.change_names_pd(data_raw, tag_dict)
        data_change_dtype = PipelineIntermediate.change_dtype_pd(data_change_names, tag_dict)
        data_intermediate = PipelineIntermediate.delete_accents_pd(data_change_dtype)

        intermediate_data_path = os.path.join(
            data_intermediate_directory,
            parameters['parameters_catalog']['intermediate_data_path'])
        Utils.save_data(data_intermediate, intermediate_data_path)

    # 3. Pipeline Primary
    @staticmethod
    def run_pipeline_primary():
        from .pipelines.primary import PipelinePrimary

        intermediate_data_path = os.path.join(
            data_intermediate_directory,
            parameters['parameters_catalog']['intermediate_data_path'])

        data_intermediate = Utils.load_data(intermediate_data_path)

        data_categorize = PipelinePrimary.recategorize_pd(data_intermediate,
                                                          parameters['parameters_primary'])

        data_impute_missing = PipelinePrimary.impute_missing_values_pd(data_categorize,
                                                                       parameters['parameters_primary'])

        data_remove_duplicates = PipelinePrimary.remove_duplicates_pd(data_impute_missing,
                                                                      parameters['parameters_primary'])

        data_primary = PipelinePrimary.remove_outliers_pd(data_remove_duplicates)

        primary_data_path = os.path.join(data_primary_directory,
                                         parameters['parameters_catalog']['primary_data_path'])
        Utils.save_data(data_primary, primary_data_path)

    # 4. Pipeline Feature
    @staticmethod
    def run_pipeline_feature():
        from .pipelines.feature import PipelineFeature

        primary_data_path = os.path.join(
            data_primary_directory,
            parameters['parameters_catalog']['primary_data_path'])

        data_primary = Utils.load_data(primary_data_path)

        data_features_new = PipelineFeature.features_new_pd(data_primary,
                                                            parameters['parameters_feature'])

        data_scaler = PipelineFeature.min_max_scaler_pd(data_features_new)

        data_feature = PipelineFeature.one_hot_encoding_pd(data_scaler,
                                                           parameters['parameters_feature'])

        feature_data_path = os.path.join(data_feature_directory,
                                         parameters['parameters_catalog']['feature_data_path'])
        Utils.save_data(data_feature, feature_data_path)

    # 5. Pipeline Models
    @staticmethod
    def run_pipeline_models():
        from .pipelines.models import PipelineModels

        feature_data_path = os.path.join(
            data_feature_directory,
            parameters['parameters_catalog']['feature_data_path'])

        data_feature = Utils.load_data(feature_data_path)

        data_kmeans = PipelineModels.train_kmeans_pd(data_feature,
                                                     parameters['parameters_models'])

        data_gmm = PipelineModels.train_gmm_pd(data_feature,
                                               parameters['parameters_models'])

        data_dbscan = PipelineModels.train_dbscan_pd(data_feature,
                                                     parameters['parameters_models'])

        models_data_path = os.path.join(data_models_directory,
                                        parameters['parameters_catalog']['models_data_path'])

        Utils.save_pickle([data_kmeans, data_gmm, data_dbscan], models_data_path)

    # 6. Pipeline Model Selection
    @staticmethod
    def run_pipeline_model_selection():
        from .pipelines.model_selection import PipelineModelSelection

        models_data_path = os.path.join(data_models_directory,
                                        parameters['parameters_catalog']['models_data_path'])

        data_models = Utils.load_pickle(models_data_path)

        feature_data_path = os.path.join(data_feature_directory,
                                         parameters['parameters_catalog']['feature_data_path'])

        data_feature = Utils.load_data(feature_data_path)

        data_select_best_model = PipelineModelSelection.select_best_model_pd(data_models,
                                                                             parameters['parameters_model_selection'])

        data_model_selection = PipelineModelSelection.optimize_train_best_gmm_pd(data_feature,
                                                                                 parameters['parameters_model_selection'])

        model_selection_data_path = os.path.join(data_model_selection_directory,
                                                 parameters['parameters_catalog']['model_selection_data_path'])

        Utils.save_pickle(data_model_selection, model_selection_data_path)

