import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

% matplotlib tk


# %%

df = pd.read_parquet("data/03_primary/train_data.parquet")

df.head()

data = df.copy()

# %%

data.info()
data.duplicated().sum()

# %%
# Limpieza general de dataset

data['age'] = data['age'].astype(str)


def clean_age(x: str) -> str:
    # Si el valor contiene solo letras, o espacios, o guiones, lo reemplaza por '0.00'
    if x.isalpha() or ' ' in x or '-' in x:
        return '0.00'
    return x

data['age'] = data['age'].apply(clean_age).astype('float32')

# %%

data['gender'] = data['gender'].map({'F': 0, 'M': 1})

# %%
data.info()

# %%
# PARTICIÓN DE DATOS
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print("Dimensiones del conjunto de entrenamiento:", train_data.shape)
print("Dimensiones del conjunto de prueba:", test_data.shape)


# %%

def fit_imputar_genero_ponderado(df: pd.DataFrame, column: str) -> pd.DataFrame:
    probabilidades = df[column].value_counts(normalize=True)
    probabilidades_dict = probabilidades.to_dict()
    return probabilidades_dict


# %%
def transform_imputar_genero_ponderado(df, col, freq_dict):
    """
    Imputa los valores nulos de la columna 'col' 
    usando la distribución dada en 'freq_dict' (por ejemplo {0:0.3, 1:0.7}).
    """
    mask_null = df[col].isnull()

    if mask_null.any():
        possible_values = list(freq_dict.keys())
        probabilities = [freq_dict[val] for val in possible_values]   
        n_nulls = mask_null.sum()
        random_choices = np.random.choice(possible_values, size=n_nulls, p=probabilities)  
        df.loc[mask_null, col] = random_choices
    return df

# %%
# IMPUTACIÓN DE GÉNERO
probabilidades_genero_train = fit_imputar_genero_ponderado(train_data, 'gender')






train_data = transform_imputar_genero_ponderado(train_data, 'gender', probabilidades_genero_train)


test_data = transform_imputar_genero_ponderado(test_data, 'gender', probabilidades_genero_train)


# %%
def fit_outlier_mzscore(df: pd.DataFrame, col: str, threshold: float = 3.5) -> dict:
    """Calcula median y MAD de la columna para usar en z-score modificado."""
    data = df[col].dropna().values
    median_val = np.median(data)
    mad_val = np.median(np.abs(data - median_val))
    if mad_val == 0:
        mad_val = 1e-9
    return {"median": median_val, "mad": mad_val, "threshold": threshold}

# %%

params_mzscore_age_train = fit_outlier_mzscore(train_data, 'age')


# %%
def transform_outlier_mzscore(df: pd.DataFrame, col: str, params: dict) -> pd.DataFrame:
    """Reemplaza outliers con la median calculada en fit_outlier_mzscore."""
    df = df.copy()
    data = df[col].values
    z_score = 0.6745 * (data - params["median"]) / params["mad"]
    outliers_mask = np.abs(z_score) > params["threshold"]
    df.loc[outliers_mask, col] = params["median"]
    return df


train_data = transform_outlier_mzscore(train_data, col="age", params=params_mzscore_age_train)
test_data = transform_outlier_mzscore(test_data, col="age", params=params_mzscore_age_train)

# %%

plt.hist(train_data['age'])

plt.hist(train_data['gender'])

# %%

# from sklearn.preprocessing import PowerTransformer

cols_to_transform = train_data.columns[3:]

# power_transformer = PowerTransformer(method='yeo-johnson')


#power_transformer.fit(train_data[cols_to_transform])

#train_data[cols_to_transform] = power_transformer.transform(train_data[cols_to_transform])

#test_data[cols_to_transform] = power_transformer.transform(test_data[cols_to_transform])



import numpy as np
from scipy import stats



for column in data.columns:
    stat, p_value = stats.shapiro(data[column])
    print(column)
    print("Estadístico de Shapiro-Wilk:", stat)
    print("p-valor:", p_value)

    if p_value > 0.05:
        print("No se rechaza la hipótesis nula: los datos parecen normales.")
    else:
        print("Se rechaza la hipótesis nula: los datos no parecen normales.")



# Luego del análisis de Yeo-Johnson, se descarta la opción de normalización de variables. En ninún caso se logra normalizar según Shapiro-Wilk.

# %%

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

cols_standard = ['age']  # variables que ya tienen distribución normal
cols_minmax = train_data.columns[3: ]            # variables que necesitan normalizarse al rango [0, 1]

scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

scaler_standard.fit(train_data[cols_standard])
scaler_minmax.fit(train_data[cols_minmax])

train_data[cols_standard] = scaler_standard.transform(train_data[cols_standard])
train_data[cols_minmax] = scaler_minmax.transform(train_data[cols_minmax])


test_data[cols_standard] = scaler_standard.transform(test_data[cols_standard])
test_data[cols_minmax] = scaler_minmax.transform(test_data[cols_minmax])

print(train_data.head())


