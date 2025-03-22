import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from pycaret.clustering import load_model, predict_model

import sklearn.impute
if not hasattr(sklearn.impute.SimpleImputer, 'verbose'):
    sklearn.impute.SimpleImputer.verbose = None


MODEL_PATH = "/app/data/06_models/best_model_birch"
fe_params = pd.read_pickle("/app/data/04_feature/fe_params.pkl")
scaler_params = pd.read_pickle("/app/data/04_feature/scalers_params.pkl")
selected_columns = pd.read_pickle("/app/data/05_model_input/selected_feature_names.pkl")

@st.cache_resource
def load_birch_model():
    # Usa la función de pycaret para cargar el modelo
    model = load_model(MODEL_PATH)
    return model


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
    cols_minmax = df.columns[3:] # Todas las columnas a partir de la cuarta
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


def predict_with_birch(data: pd.DataFrame, model) -> pd.DataFrame:
    predictions = predict_model(model, data=data)
    return predictions



# Desarrollo de interfaz
st.title("Modelo de clustering para agrupación de estudiantes.")
st.write("Ingresa los datos para realizar la predicción:")

with st.form("prediction_form"):  
    year_input = st.number_input("Año", value=2008, step=1)
    gender_input = st.selectbox("Género", options=["M", "F"])
    age_input = st.number_input("Edad", value=16, step=1)
    friends_input = st.number_input("Número de amigos", value=10, step=1)
    
    feature_names = [
         "basketball", "football", "soccer", "softball", "volleyball",
         "swimming", "cheerleading", "baseball", "tennis", "sports",
         "cute", "sex", "sexy", "hot", "kissed", "dance", "band", "marching", "music",
         "rock", "god", "church", "jesus", "bible", "hair", "dress", "blonde", 
         "mall", "shopping", "clothes",
         "hollister", "abercrombie", "die", "death", "drunk", "drugs"
    ]
    
    additional_features = {}
    st.write("Ingresa los valores para las siguientes variables:")
    for feat in feature_names:
         additional_features[feat] = st.number_input(feat.capitalize(), value=0, step=1)
    
    submitted = st.form_submit_button("Predecir")

if submitted:
    gender_numeric = 1 if gender_input == "M" else 0

    data = {
        "gradyear": [age_input],
        "gender": [gender_numeric],
        "age": [age_input],
        "NumberOffriends": [friends_input]
    }
    for feat in feature_names:
        data[feat] = [additional_features[feat]]
    df_input = pd.DataFrame(data)
    st.write("Datos ingresados:")
    st.dataframe(df_input)

    df_input = clean_df(df_input, cols=['age', 'gender'])


    df_input = transform_feature_engineering(df_input, fe_params) 
    df_input = transform_scalers(df_input, scaler_params)
    df_input = remove_gradyear(df_input)
    df_input = transform_variance_threshold(df_input, selected_columns)
    model = load_birch_model()
    prediction = predict_model(model, data=df_input)
    st.write("Resultado de la predicción:")
    output = prediction.iloc[0].get("Cluster", None)
    if output is not None:
        st.write(output)
    else:
        st.write("No se encontró la columna 'Cluster' en el output.")

#	st.dataframe(prediction)
