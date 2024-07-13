import os
from .utils import Utils
from .pipelines.raw import PipelineRaw
from .pipelines.intermediate import PipelineIntermediate

logger = Utils.setup_logging()
Utils.add_src_to_path()
project_root = Utils.get_project_root()

parameters_directory = os.path.join(project_root, 'src', 'parameters')
data_raw_directory = os.path.join(project_root, 'data', '01_raw')
data_intermediate_directory = os.path.join(project_root, 'data', '02_intermediate')

parameters = Utils.load_parameters(parameters_directory)


class PipelineOrchestration:

    # 1. Pipeline Raw
    @staticmethod
    def run_pipeline_raw():
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







