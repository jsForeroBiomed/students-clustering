# %%
## INTRODUCCIÓN

# %%
# En este archivo se implementa el Análisis Exploratorio de Variables para el dataset de agrupación de estudiantes. 


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
## CARGA Y EXPLORACIÓN DE LOS DATOS

# %%
data = pd.read_csv("data/01_raw/01_raw_clustering_analysis_of_students.csv")
data.head()
# El dataset cuenta con 40 atributos y 15000 instancias. 

# %%
## EXPLORACIÓN INICIAL DE LOS DATOS

# %%
data.columns.values
data.info()
# 38 variables son enteras (int64) y dos se reconocen como objeto.
# Variables reconocidas como objeto
# * age
# * gender
# Las 38 variables numéricas tienen 15000 instancias. 
# Para `age` se cuenta con 12504 instancias. 
# Para `gender` se cuenta con 13663 instancias. 

# %%
duplicados = data.duplicated().sum()
data.info()
print(duplicados)
# El dataset cuenta con 266 filas duplicadas

# %%
datos_faltantes = data.isnull().sum()
datos_faltantes_porcentaje = ( data.isna().sum() / len(data) ) * 100
print(datos_faltantes, datos_faltantes_porcentaje)
# Variables con valores faltantes: `gender` y `age`
# Valores faltantes en `gender` - 1337 (7.73%)
# Valores faltantes en Edad - 2496 (15.21%)

# %%
data.nunique()
# Las variables numéricas son en su mayoría enteras y tienen hasta 25 valores diferentes. 
# La únicas variables que tienen muchos más valores diferentes son `age` y `NumberOffriends`
# `age` tiene 1906 valores diferentes. 
# `NumberOffriends` tiene 236


# %%
# A modo de estandarización, se recomienda renombrar la variable de "NumberOffriends". 

# %%
# Se importante recalcar que `age` se encuentra registrada como variable flotante. 
# Se puede calcular en inferencias como edad = dias_desde_nacimiento / 365 para obtener el valor glorante. 

# %%
## ANÁLISIS DE VALORES FALTANTES
# Se recuerda que las dos variables con valores faltantes son `age` y `gender`.
# Para poder analizar los outliers de 'age', se debe modificar la variable. 
# Se crea una copia de la variable en una variable `age_for_outliers`
age_for_outliers = data['age'].copy()
age_for_outliers = age_for_outliers.apply(lambda x: str(x))
age_for_outliers = age_for_outliers.apply(lambda x: '0.00' if x.isalpha() else x)
age_for_outliers = age_for_outliers.apply(lambda x: '0.00' if ' ' in x or '-' in x else x).astype('float32')

# Se convierten todos los valores atípicos en 0.00 para poder analizar el componente numérico de la variable. 

# %%
# Para el análisis de `gender`, se analiza primero la distribución del atributo. 
gender_value_counts = data['gender'].value_counts()

# Se evidencia que hay desbalance entre género masculino y femenino en la clase donde la distribución se ve así:
# * F: 11001
# * M: 2593
# Además, se recomienda convertir la variable `gender` a binaria.
# gender_mapping = data['gender'].map({'F': 0, 'M': 1})

# %%

def imputar_genero_ponderado(df: pd.DataFrame, column: str) -> pd.DataFrame:
    probabilidades = df[column].value_counts(normalize=True)
    df[column] = df[column].apply(lambda x: np.random.choice(probabilidades.index, p=probabilidades.values) if pd.isna(x) else x)
    return df


# %% ANÁLISIS DE DISTRIBUCIÓN DE CADA VARIABLE
# Histogramas de variables enteras.
for column in data.columns:
    if data[column].dtype == np.int64:
        plt.hist(data[column])
        plt.savefig(f'data/08_reporting/histograma_{column}.png')
        plt.close()
        
        plt.boxplot(data[column])
        plt.savefig(f'data/08_reporting/boxplot_{column}.png')
        plt.close()

# Para las variables enteras, la mayoría muestra un comportamiento de cola larga, con mayor densidad en los valores bajos y menor en valores altos.
# El comportamiento cambia únicamente para el `gradyear` donde todos los años muestran una distribución uniforme. 


# Histogramas de edad
plt.hist(age_for_outliers[age_for_outliers > 0])
plt.savefig(f'data/08_reporting/histograma_age.png')
plt.close()
plt.boxplot(age_for_outliers[age_for_outliers > 0])
plt.savefig(f'data/08_reporting/boxplot_age.png')
plt.close()

# Para la edad, se evidencia que hay una gran cantidad de outliers, con valores de edad tanto de cero años como de cien años. Se recomienda definir un rango adecuado de variables e imputar los valores fuera del rango.

# %%
## ANÁLISIS DE OUTLIERS
# Para el análisis de outliers, los valores evidenciados en las variables enteras parecen ser valores dentro de lo esperado por lo que no se recomendarían eliminar, incluso si son valores atípicos. 

# Para el caso de la edad, se implementa un análisis más a detalle de la variable, para cuantificar el número de outliers. Para esto, se utiliza el Z-Score modificado, dado que el histograma de la edad muestra una distribución no normal. 

def compute_modified_z_score(data, threshold=3.5):
    data = np.array(data)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_score = 0.6745 * (data - median) / (mad if mad else 1e-9)  # Evita división por cero
    return modified_z_score

def detect_outliers_with_modified_z_score(data, modified_z_score, threshold=3.5):
    outliers = data[np.abs(modified_z_score) > threshold]
    return outliers

# def replace_outliers_with_median(data, col, modified_z_score, threshold=3.5):
#    data.loc[np.abs(modified_z_score) > threshold, col] = np.median(data[col])
#    return data


# plt.hist(data['gradyear'])
# En `gender` hay varios outliers de valores por encima de 100 y otros con 0.

# %%
# Cálculo de Z score modificado para cada valor de la edad
modified_z_score = compute_modified_z_score(age_for_outliers[age_for_outliers > 0])

age_outliers = detect_outliers_with_modified_z_score(age_for_outliers[age_for_outliers > 0],
                                                     modified_z_score)

# Mediante el cálculo de los outliers para la edad se identifican 158 valores atípicos.
# En el cálculo se omiten los valores con error de digitación que se igual a 0.00 previamente. 
# Mediana de outliers: 95.78 años
# Outlier mínimo: 4.31 años
# Outlier máximo: 106.92 años

# Por el contexto del problema, dado que se está haciendo la agrupación entre estudiantes, se pueden imputar los valores atípicos detectados por Z-Score
# 158 outliers de 12228 valores de edad, representan un 1.29% del dataset. 


# %%
# ANÁLISIS DE CORRELACIONES
# Para poder graficar, se debe modificar las variables. Se genera una copia del dataset original parañadir las variables modificadas que simplifiquen el análisis inicial de correlación. 

correlation_data = data.copy()
# Para el análisis rápido de los datos de correlación, se eliminan los valores nan de la variable `gender`
correlation_data.dropna(inplace=True)

mapa_genero = {'M': 1, 'F': 0}
correlation_data['gender'] = correlation_data['gender'].map(mapa_genero)
correlation_data['age'] = age_for_outliers

# %%

# Pearson
plt.figure(figsize=(12, 10))
x = sns.heatmap(correlation_data[correlation_data['age'] > 0].corr(), annot=True, cmap='crest', fmt='.1f', annot_kws={"size": 8})
plt.xticks(fontsize=8, rotation=45, ha='right')  # Alinea a la derecha
plt.yticks(fontsize=8, rotation=0)

plt.savefig(f'data/08_reporting/matriz_correlacion_pearson.png')

# Al hacer el análisis de correlación por Pearson, no se evidencia ninguna correlación superior a 0.8 como para considerar una redundancia en la variable. 
# Sin embargo, se encuentran algunas variables que muestran símbolos de correlación entre ellas. 

# Hollister y Abercrombie - Correlación de 0.5. Tiene sentido al ser ambas marcas de ropa
# Marching y Band - Correlación de 0.5. Marchas y Bandas. Podría haber relación entre ellas. 

# %%
# Spearman
plt.figure(figsize=(12, 10))
x = sns.heatmap(correlation_data[correlation_data['age'] > 0].corr(method='spearman'), annot=True, cmap='crest', fmt='.1f', annot_kws={"size": 8})
plt.xticks(fontsize=8, rotation=45, ha='right')  # Alinea a la derecha
plt.yticks(fontsize=8, rotation=0)

plt.savefig(f'data/08_reporting/matriz_correlacion_spearman.png')

# Para el caso de Spearman se evidencian correlaciones más fuertes entre algunas variables. 
# El año de graduación y la edad están muy correlacionadas de manera inversa (-0.9). Al ser ambas variables similares, se podría eliminar el año de graduación que contiene menos información. 
# En Spearman también se observa una correlación ligera (0.5) entre Abercrombie y Hollister.

# %%
## CONCLUSIÓN DE EDA
# * Las variables que cuentan con valores faltantes son Edad y Género, ambas en proporciones bajas por lo que se podría hacer imputación. 
# * Se evidencia desbalance entre género Femenino y Masculino, siendo el primero mucho más representativo en la base de datos.
# * Se evidencia que todas las variables tienen distribución cola larga similar. Se podría validar si una opción sería transformarlas logarítmicamente. 
# * Para outliers, se considera que solo se debería tratar la edad, dado que los valores atípicos de las demás variables parecen normales, y de esperar en deploy. 
# * Se identifica una correlación muy alta entre Edad y Año de Graduación, lo que muestra que se podría considerar la eliminación del Año de Graduación. Además, otras variables como Abercrombie y Hollister muestra ligera correlación. 


