from studentsproject.pipelines.data_partitioning.pipeline import create_pipeline as create_data_partitioning_pipeline
from studentsproject.pipelines.feature_engineering.pipeline import create_pipeline as create_fe_pipeline
from studentsproject.pipelines.feature_selection.pipeline import create_pipeline as create_fs_pipeline
from studentsproject.pipelines.model_inference.pipeline import create_pipeline as create_model_inf_pipeline


def register_pipelines():
    """
    Registra y retorna todos los pipelines del proyecto.
    """
    pipelines = {
        "data_partitioning": create_data_partitioning_pipeline(),
        "feature_engineering": create_fe_pipeline(),
        "feature_selection": create_fs_pipeline(),
        "model_inference": create_model_inf_pipeline(),
        "__default__": create_data_partitioning_pipeline()  # El pipeline por defecto
    }
    return pipelines

