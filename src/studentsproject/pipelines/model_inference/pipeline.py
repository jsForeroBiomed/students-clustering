from kedro.pipeline import Pipeline, node
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_df,
                inputs="test_data",  
                outputs="clean_test_data",
                name="clean_test_data_node"
                ),
            node(
                func=transform_feature_engineering,
                inputs=["clean_test_data", "fe_params"], 
                outputs="engineered_test_data",
                name="transform_feature_engineering_test_node"
                ),
            node(
                func=transform_scalers,
                inputs=["engineered_test_data", "scalers_params"],
                outputs="scaled_test",
                name="transform_scalers_test_node"
                ),	
            node(
                func=remove_gradyear,
                inputs="scaled_test", 
                outputs="fs_scaled_test",
                name="remove_gradyear_test_node"
                ),
            node(
                func=transform_variance_threshold,
                inputs=["fs_scaled_test", "selected_feature_names"],
                outputs="selected_test_data",
                name="transform_variance_threshold_node"
                ),
            node(
                func=predict_with_birch,
                inputs="selected_test_data",  
                outputs="pycaret_predictions",
                name="pycaret_prediction_node"
            )
        ]
    )

