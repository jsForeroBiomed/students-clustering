from kedro.pipeline import Pipeline, node
from .nodes import remove_duplicates, partition_data

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=remove_duplicates,
                inputs="raw_data",  # Este es el dataset definido en catalog.yml
                outputs="deduplicated_data",
                name="remove_duplicates_node"
            ),
            node(
                func=partition_data,
                inputs="deduplicated_data",
                outputs=["train_data", "test_data"],
                name="partition_data_node"
            )
        ]
    )

