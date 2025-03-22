import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def remove_gradyear(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina la columna 'gradyear' del DataFrame, asumiendo que se elimina por alta correlación.
    """
    df_new = df.copy()
    if "gradyear" in df_new.columns:
        df_new = df_new.drop(columns=["gradyear"])
    return df_new


def fit_variance_threshold(df: pd.DataFrame, threshold: float = 0.001) -> tuple:
    """
    Aplica VarianceThreshold al DataFrame para eliminar variables de baja varianza.
    
    Args:
        df: DataFrame de entrada (se espera que ya esté limpio, por ejemplo, sin 'gradyear').
        threshold: Umbral de varianza para mantener una variable.
    
    Returns:
        tuple: (df_reduced, selected_columns)
            - df_reduced: DataFrame resultante con solo las columnas que superan el umbral.
            - selected_columns: Lista de nombres de columnas seleccionadas.
    """
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(df)
    selected_columns = df.columns[vt.get_support()]
    # Transformamos el DataFrame y lo convertimos a DataFrame (manteniendo índice y nombres)
    df_reduced = pd.DataFrame(vt.transform(df), columns=selected_columns, index=df.index)
    return df_reduced, list(selected_columns)


def transform_variance_threshold(df: pd.DataFrame, selected_columns: list) -> pd.DataFrame:
    """
    Selecciona del DataFrame únicamente las columnas indicadas en selected_columns.
    
    Args:
        df: DataFrame de entrada.
        selected_columns: Lista de columnas a conservar.
    
    Returns:
        DataFrame con las columnas seleccionadas.
    """
    return df[selected_columns].copy()

