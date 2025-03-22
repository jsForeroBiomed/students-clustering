# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
%matplotlib tk

# %%
df_train = pd.read_parquet("data/04_feature/scaled_internal_train")
df_val = pd.read_parquet("data/04_feature/scaled_internal_val")

# %%

df_train.corr()
# Por correlación alta entre `age` y `gradyear` se elimina `gradyear`.
# %%

data_train = df_train.copy()
data_val = df_val.copy()

del data_train['gradyear']

del data_val['gradyear']

# %%
from sklearn.feature_selection import VarianceThreshold


data_train_columns = data_train.columns

variances = data_train.var()
print(variances)


vt = VarianceThreshold(threshold=0.001)
vt.fit(data_train[data_train_columns])
data_train_reduced = vt.transform(data_train[data_train_columns])
data_train_reduced = vt.transform(data_train[data_train_columns])

columns_selected = data_train_columns[vt.get_support()]

print(list(columns_selected))
