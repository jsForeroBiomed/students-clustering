# %%
import pandas as pd
from pycaret.clustering import *
import matplotlib.pyplot as plt

%matplotlib tk

# %%
data = pd.read_parquet("data/05_model_input/selected_train_data.parquet")

# %%
exp = setup(data=data, normalize=False, session_id=123)


# %%
kmeans_model = create_model('kmeans', num_clusters=3)

# %%

plot_model(kmeans_model, plot='cluster')
# plt.savefig("data/")


# %%
results = {}

for name, model_id in model_ids.items():
    try:
        print(f"Creando y evaluando {name}...")
        model = create_model(model_id)
        evaluate_model(model)
		metrics_df = pull()
		results[model_id] = metrics_df
    except Exception as e:
        print(f"Error al crear {name}: {e}")


# save_model(model, model_name="best_clustering_model")

