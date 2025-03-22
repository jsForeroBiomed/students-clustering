import pickle
import pandas as pd
from pathlib import Path

# Cargar el modelo
MODEL_PATH = Path(__file__).resolve().parent.parent / "data/06_models/best_model_birch.pkl"
COLUMNS_PATH = Path(__file__).resolve().parent.parent / "data/06_models/selected_feature_names.pkl"


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(COLUMNS_PATH, "rb") as f:
    feature_names = pickle.load(f)

def predict(input_data):
    """Recibe un diccionario con las características y devuelve la predicción."""
    df = pd.DataFrame([input_data], columns=feature_names)
    return model.predict(df)

