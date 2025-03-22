from kedro.pipeline import Pipeline, node
from .nodes import (
    remove_gradyear,
    fit_variance_threshold,
    transform_variance_threshold
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # 1. Eliminar 'gradyear' en el conjunto de entrenamiento
            node(
                func=remove_gradyear,
                inputs="scaled_internal_train",
                outputs="fs_train_temp",
                name="remove_gradyear_train_node"
            ),
            # 2. Eliminar 'gradyear' en el conjunto de validación
            node(
                func=remove_gradyear,
                inputs="scaled_internal_val",
                outputs="fs_val_temp",
                name="remove_gradyear_val_node"
            ),
            # 3. Ajustar VarianceThreshold en training (fit)
            node(
                func=fit_variance_threshold,
                inputs="fs_train_temp",
                outputs=["selected_train_data", "selected_feature_names"],
                name="fit_variance_threshold_node"
            ),
            # 4. Transformar el conjunto de validación usando las variables seleccionadas
            node(
                func=transform_variance_threshold,
                inputs=["fs_val_temp", "selected_feature_names"],
                outputs="selected_val_data",
                name="transform_variance_threshold_node"
            )
        ]
    )

