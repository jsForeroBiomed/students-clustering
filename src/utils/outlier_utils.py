import numpy as np
import pandas as pd

def compute_modified_z_score(data: np.ndarray) -> np.ndarray:
    """
    Calcula el Modified Z-score para un array de valores numéricos.
    
    Args:
        data (np.ndarray): Array unidimensional de valores numéricos.

    Returns:
        np.ndarray: Array con los valores del Modified Z-score para cada elemento de 'data'.
    """
    data_array = np.array(data)
    median_val = np.median(data_array)
    mad = np.median(np.abs(data_array - median_val))
    # Evita división por cero si el MAD es 0
    modified_z = 0.6745 * (data_array - median_val) / (mad if mad else 1e-9)
    return modified_z


def detect_outliers_with_modified_z_score(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    Retorna una máscara booleana indicando cuáles valores en 'data' son atípicos
    según el Modified Z-score.
    
    Args:
        data (np.ndarray): Array unidimensional de valores numéricos.
        threshold (float, optional): Umbral para considerar un valor como atípico. 
                                     Por defecto, 3.5.

    Returns:
        np.ndarray: Array booleano de la misma longitud que 'data', con True en las 
                    posiciones que representan valores atípicos.
    """
    z_scores = compute_modified_z_score(data)
    outliers_mask = np.abs(z_scores) > threshold
    return outliers_mask


def replace_outliers_with_median(df: pd.DataFrame, col: str, threshold: float = 3.5) -> pd.DataFrame:
    """
    Reemplaza los valores atípicos de la columna 'col' en el DataFrame 'df' 
    con la mediana de esa columna, usando el Modified Z-score.
    
    Args:
        df (pd.DataFrame): DataFrame con la columna a procesar.
        col (str): Nombre de la columna en la cual se detectarán y reemplazarán atípicos.
        threshold (float, optional): Umbral para considerar un valor como atípico. 
                                     Por defecto, 3.5.

    Returns:
        pd.DataFrame: El mismo DataFrame 'df' con los atípicos reemplazados por la mediana.
    """
    outliers_mask = detect_outliers_with_modified_z_score(df[col].values, threshold)
    median_val = df[col].median()
    df.loc[outliers_mask, col] = median_val
    return df

