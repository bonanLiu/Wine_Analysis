import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Import the processed dataframe from Regression.py
from Regression_process import (df_red,df_white)

df_red['quality_new'] = np.where(df_red['quality'].between(1, 5), 0, 1)
df_white['quality_new'] = np.where(df_white['quality'].between(1, 5), 0, 1)

print(df_red.shape,df_white.shape)