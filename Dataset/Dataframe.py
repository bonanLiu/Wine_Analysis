import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sb
import matplotlib.pyplot as plt



# 1. Create two dataframe for Red and White

df_red=pd.read_csv("Dataset/wine_quality_Red.csv")
df_white=pd.read_csv("Dataset/wine_quality_White.csv")

# print(df_red.info())
# print(df_white.info())

# 2. basoc information
# print(df_red.shape,df_white.shape)
# print(df_red.describe(),df_white.describe())


# 2. Correlation with Pearson and Heatmap

#  Red 
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



# Two Heatmaps in One page
# Create Correlation Matrix
red_pear=df_red.corr()
white_pear=df_white.corr()

# Layout Set
fig, axes=plt.subplots(1, 2, figsize=(20, 8)) 

# Heatmap
sb.heatmap(red_pear, annot=True, cmap='inferno', fmt='.2f', linewidths=0.5, cbar=True, ax=axes[0])
axes[0].set_title('Correlation Heatmap of Red Wine Features')

sb.heatmap(white_pear, annot=True, cmap='inferno', fmt='.2f', linewidths=0.5, cbar=True, ax=axes[1])
axes[1].set_title('Correlation Heatmap of White Wine Features')

# Result
plt.tight_layout()
plt.show()


# 3. Histograms for Selected Variables

Red_to_plot=['fixed acidity', 'citric acid', 'residual sugar', 'sulphates', 'alcohol','quality']
White_to_plot=['free sulfur dioxide', 'pH', 'sulphates', 'alcohol','quality']

# Layout Set and Histograms
fig, axes=plt.subplots(2,6,figsize=(20, 8)) 

for i, column in enumerate(Red_to_plot):
    sb.histplot(df_red[column], kde=True, bins=10, color="skyblue", edgecolor="black",ax=axes[0, i])
    plt.title(f'Red Wine Distribution of {column}')

for i, column in enumerate(White_to_plot):
    sb.histplot(df_white[column], kde=True, bins=10, color="skyblue", edgecolor="black",ax=axes[1, i])
    plt.title(f'White Wine Distribution of {column}')

for j in range(len(White_to_plot), 6):
    fig.delaxes(axes[1, j])

plt.tight_layout()
plt.show()


# 4. Find Outliers





