import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import zscore



# 1. Create two dataframe for Red and White

df_red=pd.read_csv("Dataset/wine_quality_Red.csv")
df_white=pd.read_csv("Dataset/wine_quality_White.csv")

# print(df_red.info())
# print(df_white.info())

# 2. basic information
# print(df_red.shape,df_white.shape)
# print(df_red.describe(),df_white.describe())

# 3. check data

# print(df_red.isnull().sum())
# print(df_white.isnull().sum())

# print(df_red.isna().sum())
# print(df_white.isna().sum())

# print(df_red.duplicated())
# print(df_white.duplicated())

df_red=df_red.drop_duplicates()
df_white=df_white.drop_duplicates()

# print(df_red.shape,df_white.shape)


### Descriptive Analysis

# 4. Correlation with Pearson and Heatmap

# #  Red 
# red_pear=df_red.corr()
# plt.figure(figsize=(10, 8))
# sb.heatmap(red_pear, annot=True, cmap='inferno', fmt='.2f', linewidths=0.5, cbar=True)
# plt.title('Correlation Heatmap of Red Wine Features')


# #  White
# white_pear=df_white.corr()
# plt.figure(figsize=(10, 8))
# sb.heatmap(white_pear, annot=True, cmap='inferno', fmt='.2f', linewidths=0.5, cbar=True)
# plt.title('Correlation Heatmap of White Wine Features')
# plt.show()



# # Two Heatmaps in One page
# # Create Correlation Matrix
# red_pear=df_red.corr()
# white_pear=df_white.corr()

# # Layout Set
# fig, axes=plt.subplots(1, 2, figsize=(20, 8)) 

# # Heatmap
# sb.heatmap(red_pear, annot=True, cmap='inferno', fmt='.2f', linewidths=0.5, cbar=True, ax=axes[0])
# axes[0].set_title('Correlation Heatmap of Red Wine Features')

# sb.heatmap(white_pear, annot=True, cmap='inferno', fmt='.2f', linewidths=0.5, cbar=True, ax=axes[1])
# axes[1].set_title('Correlation Heatmap of White Wine Features')

# # Result
# plt.tight_layout()
# plt.show()


# 5. Histograms for Selected Variables

Red_to_plot=['fixed acidity', 'citric acid', 'residual sugar', 'sulphates', 'alcohol','quality']
White_to_plot=['citric acid', 'free sulfur dioxide', 'pH', 'sulphates', 'alcohol','quality']

# Layout Set and Histograms
fig, axes=plt.subplots(2,6,figsize=(20, 10)) 


# Red wine histogram 
for i, column in enumerate(Red_to_plot):
    ax = axes[0, i]
    sb.histplot(df_red[column], kde=True, bins=10, color="skyblue", edgecolor="black", ax=ax)
    ax.set_title(f"Red_{column}", fontsize=12)
    ax.set_xlabel('')

# White wine histogram 
for i, column in enumerate(White_to_plot):
    ax = axes[1, i] 
    sb.histplot(df_white[column], kde=True, bins=10, color="skyblue", edgecolor="black", ax=ax)
    ax.set_title(f"White_{column}", fontsize=12)
    ax.set_xlabel('')

plt.tight_layout()
plt.show()


# 6. Find Outliers - Zscore for all features

Zscore_red=zscore(df_red[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                        'pH', 'sulphates', 'alcohol']])
outliers_red=(abs(Zscore_red)>3).sum(axis=0)
print(outliers_red)




