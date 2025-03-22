import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def clean_age(x: str) -> str:
    if x.isalpha() or ' ' in x or '-' in x:
        return '0.00'
    return x


def clean_df(df: pd.DataFrame, cols=['age', 'gender']) -> pd.DataFrame:
    df[cols[0]] = df[cols[0]].fillna('0')
    df[cols[0]] = df[cols[0]].astype('str')
    df[cols[0]] = df[cols[0]].apply(lambda x: '0' if x.isalpha() or ' ' in x or '-' in x else x)
    df[cols[0]] = df[cols[0]].astype('float32')
    df[cols[1]] = df[cols[1]].map({'F': 0, 'M': 1})
    return df    


def partition_train_val(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Separa el conjunto de training en dos subconjuntos: 
    'train' (para ajuste) e 'internal_val' (para validación interna).
    """
    train, internal_val = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, internal_val


def fit_outlier_mzscore(df: pd.DataFrame, col: str = 'age', threshold: float = 3.5) -> dict:
    data = df[col].dropna().values
    median_val = np.median(data)
    mad_val = np.median(np.abs(data - median_val))
    if mad_val == 0:
        mad_val = 1e-9
    return {"median": median_val, "mad": mad_val, "threshold": threshold}


def transform_outlier_mzscore(df: pd.DataFrame, params: dict, col: str = 'age') -> pd.DataFrame:
    df = df.copy()
    data = df[col].values
    z_score = 0.6745 * (data - params["median"]) / params["mad"]
    outliers_mask = np.abs(z_score) > params["threshold"]
    df.loc[outliers_mask, col] = params["median"]
    return df


def fit_imputar_genero_ponderado(df: pd.DataFrame, col: str = 'gender') -> dict:
    """Calcula la distribución (0 y 1) en 'gender' y retorna un dict, ej. {0:0.3, 1:0.7}."""
    return df[col].value_counts(normalize=True, dropna=True).to_dict()


def transform_imputar_genero_ponderado(df: pd.DataFrame, distrib: dict,col: str = 'gender') -> pd.DataFrame:
    """Imputa los valores nulos en 'gender' usando la distribución proporcionada."""
    df = df.copy()
    mask = df[col].isnull()
    if mask.any():
        values = list(distrib.keys())
        probs = [distrib[v] for v in values]
        n_nulls = mask.sum()
        df.loc[mask, col] = np.random.choice(values, size=n_nulls, p=probs)
    return df



def fit_feature_engineering(df: pd.DataFrame) -> dict:
    """
    Calcula y retorna los parámetros necesarios para transformar el dataset.
    Por ejemplo:
      - Parámetros para tratar outliers en "age".
      - Distribución para imputar "gender".
    Se calcula solo con el conjunto de training.
    """
    # Calcula parámetros para outliers en "age"
    params_age = fit_outlier_mzscore(df, col="age", threshold=3.5)
    # Calcula la distribución de "gender" para imputar valores nulos
    params_gender = fit_imputar_genero_ponderado(df, col="gender")
    
    return {"age": params_age, "gender": params_gender}


def transform_feature_engineering(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Transforma el DataFrame usando los parámetros dados.
    Aplica las transformaciones (por ejemplo, tratamiento de outliers e imputación de género)
    según los parámetros calculados en la fase de fit.
    """
    df = transform_outlier_mzscore(df, params=params["age"])
    df = transform_imputar_genero_ponderado(df, distrib=params["gender"])
    return df


def fit_scalers(df: pd.DataFrame) -> dict:
    """
    Ajusta dos escaladores:
      - StandardScaler para columnas en cols_standard.
      - MinMaxScaler para columnas en cols_minmax.
    Retorna un diccionario con ambos escaladores.
    """
    cols_standard = df.columns[2] # 'age' está en la posición 2
    cols_minmax = df.columns[3: ] # Todas las columnas a partir de la posición 3
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()
    
    scaler_standard.fit(df[[cols_standard]])
    scaler_minmax.fit(df[cols_minmax])
    
    return {"standard": scaler_standard, "minmax": scaler_minmax}


def transform_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """
    Transforma el DataFrame usando los escaladores ajustados:
      - Aplica StandardScaler a cols_standard.
      - Aplica MinMaxScaler a cols_minmax.
    """
    df = df.copy()
    cols_standard = df.columns[2] # `age` está en posición 2
    cols_minmax = df.columns[3: ] # Todas las columnas a partir de la cuarta
    df[cols_standard] = scalers["standard"].transform(df[[cols_standard]])
    df[cols_minmax] = scalers["minmax"].transform(df[cols_minmax])
    return df
