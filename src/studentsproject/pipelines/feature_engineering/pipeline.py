from kedro.pipeline import Pipeline, node
from .nodes import (
    clean_df,
    partition_train_val,
    fit_feature_engineering,
    transform_feature_engineering,
    fit_scalers,
    transform_scalers
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # Limpia los datos de entrenamiento
            node(
                func=clean_df,
                inputs="train_data",
                outputs="clean_train_data",
                name="clean_train_data_node"
                ),
            # Particiona el conjunto de training en train e internal validation
            node(
                func=partition_train_val,
                inputs="clean_train_data",
                outputs=["internal_train", "internal_val"],
                name="partition_train_val_node"
            ),
            # Ajusta los parámetros de feature engineering con internal_train
            node(
                func=fit_feature_engineering,
                inputs="internal_train",
                outputs="fe_params",
                name="fit_feature_engineering_node"
            ),
            # Transforma el conjunto de internal_train
            node(
                func=transform_feature_engineering,
                inputs=["internal_train", "fe_params"],
                outputs="engineered_internal_train",
                name="transform_feature_engineering_train_node"
            ),
            # Transforma el conjunto de internal_val usando los mismos parámetros
            node(
                func=transform_feature_engineering,
                inputs=["internal_val", "fe_params"],
                outputs="engineered_internal_val",
                name="transform_feature_engineering_val_node"
            ),
            # Ajustar escaladores 
            node(
                func=fit_scalers,
                inputs=["engineered_internal_train"],
                outputs="scalers_params",
                name="fit_scalers_node"
            ),
            # Transformar escaladores en train
            node(
                func=transform_scalers,
                inputs=["engineered_internal_train", "scalers_params"],
                outputs="scaled_internal_train",
                name="transform_scalers_train_node"
            ),

            # Transforma scalers en val usando los mismos parámetros
            node(
                func=transform_scalers,
                inputs=["engineered_internal_val", "scalers_params"],
                outputs="scaled_internal_val",
                name="transform_scalers_val_node"
            ),


        ]
    )

