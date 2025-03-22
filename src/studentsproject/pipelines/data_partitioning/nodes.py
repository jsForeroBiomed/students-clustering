from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas duplicadas del DataFrame."""
    return df.drop_duplicates()

def partition_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el DataFrame en entrenamiento y prueba.
    Retorna un diccionario con claves "train" y "test".
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

