import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from pycaret.clustering import load_model, predict_model

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


def transform_outlier_mzscore(df: pd.DataFrame, params: dict, col: str = 'age') -> pd.DataFrame:
    df = df.copy()
    data = df[col].values
    z_score = 0.6745 * (data - params["median"]) / params["mad"]
    outliers_mask = np.abs(z_score) > params["threshold"]
    df.loc[outliers_mask, col] = params["median"]
    return df


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


def transform_feature_engineering(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Transforma el DataFrame usando los parámetros dados.
    Aplica las transformaciones (por ejemplo, tratamiento de outliers e imputación de género)
    según los parámetros calculados en la fase de fit.
    """
    df = transform_outlier_mzscore(df, params=params["age"])
    df = transform_imputar_genero_ponderado(df, distrib=params["gender"])
    return df


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

def remove_gradyear(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    if "gradyear" in df_new.columns:
        df_new = df_new.drop(columns=["gradyear"])
    return df_new


def transform_variance_threshold(df: pd.DataFrame, selected_columns: list) -> pd.DataFrame:
    return df[selected_columns].copy()


def predict_with_birch(data: pd.DataFrame) -> pd.DataFrame:
    model = load_model("data/06_models/best_model_birch")
    predictions = predict_model(model, data=data)
    return predictions
