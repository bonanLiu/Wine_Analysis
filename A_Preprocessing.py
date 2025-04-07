import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import zscore



###  Create two dataframe for Red and White

df_red=pd.read_csv("Dataset/wine_quality_Red.csv")
df_white=pd.read_csv("Dataset/wine_quality_White.csv")

summary_data = {
    'Dataset': ['Red Wine', 'White Wine'],
    'Shape': [f"{df_red.shape[0]} × {df_red.shape[1]}", 
              f"{df_white.shape[0]} × {df_white.shape[1]}"],
    'Missing Values': [df_red.isnull().sum().sum(), 
                       df_white.isnull().sum().sum()],
    'Duplicate Rows': [df_red.duplicated().sum(), 
                       df_white.duplicated().sum()]
}
df_summary = pd.DataFrame(summary_data)

fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')  
table = ax.table(cellText=df_summary.values,
                 colLabels=df_summary.columns,
                 loc='center',
                 cellLoc='center',
                 colLoc='center')

table.scale(1, 2) 
table.auto_set_font_size(False)
table.set_fontsize(12)

plt.tight_layout()
plt.show()


print("Before Dropping Duplicates:")
print(f"Red:{df_red.shape}\nWhite:{df_white.shape}")

df_red=df_red.drop_duplicates()
df_white=df_white.drop_duplicates()

print("\nAfter Dropping Duplicates:")
print(f"Red:{df_red.shape}\nWhite:{df_white.shape}")

##  Find Outliers - Zscore for all features

Zscore_red=zscore(df_red[['fixed acidity', 'volatile acidity', 
                          'citric acid', 'residual sugar', 
                        'chlorides', 'free sulfur dioxide', 
                        'total sulfur dioxide', 'density', 
                        'pH', 'sulphates', 'alcohol']])
outliers_red=(abs(Zscore_red)>3).sum(axis=0)

outliers_redmask = abs(Zscore_red) > 3  
outliers_reddata = df_red[outliers_redmask.any(axis=1)]  

Zscore_white=zscore(df_white[['fixed acidity', 'volatile acidity', 
                              'citric acid', 'residual sugar', 
                        'chlorides', 'free sulfur dioxide', 
                        'total sulfur dioxide', 'density', 
                        'pH', 'sulphates', 'alcohol']])
outliers_white=(abs(Zscore_white)>3).sum(axis=0)

outliers_whitemask = abs(Zscore_white) > 3  
outliers_whitedata = df_white[outliers_whitemask.any(axis=1)]  

# delete all bias
df_red = df_red[(abs(Zscore_red) <= 3).all(axis=1)]
df_white = df_white[(abs(Zscore_white) <= 3).all(axis=1)]
# # print details of bias
print(outliers_reddata)
print(outliers_whitedata)
print("\nAfter Dropping Bias:")
print(f"Red:{df_red.shape}\nWhite:{df_white.shape}")







### Descriptive Analysis

##  Correlation with Pearson and Heatmap

#  Red 
red_pear=df_red.corr()
plt.figure(figsize=(10, 8))
sb.heatmap(red_pear, annot=True, cmap='inferno', fmt='.2f', linewidths=0.5, cbar=True)
plt.title('Correlation Heatmap of Red Wine Features')


#  White
white_pear=df_white.corr()
plt.figure(figsize=(10, 8))
sb.heatmap(white_pear, annot=True, cmap='YlGn', fmt='.2f', linewidths=0.5, cbar=True)
plt.title('Correlation Heatmap of White Wine Features')
plt.show()



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


#  Histograms for Selected Variables

Red_to_plot = ['citric acid', 'sulphates', 'alcohol','fixed acidity','residual sugar', 'quality']
White_to_plot=['citric acid', 'sulphates', 'alcohol','free sulfur dioxide', 'pH','quality']

# Layout Set and Histograms
fig, axes=plt.subplots(2,6,figsize=(20, 10)) 


# Red wine histogram 
for i, column in enumerate(Red_to_plot):
    ax = axes[0, i]
    sb.histplot(df_red[column], kde=True, bins=10, color="#d0a7cf", edgecolor="black", ax=ax)
    ax.set_title(f"Red_{column}", fontsize=12)
    ax.set_xlabel('')

# White wine histogram 
for i, column in enumerate(White_to_plot):
    ax = axes[1, i] 
    sb.histplot(df_white[column], kde=True, bins=10, color="#bdddb6", edgecolor="black", ax=ax)
    ax.set_title(f"White_{column}", fontsize=12)
    ax.set_xlabel('')

plt.tight_layout()
plt.show()
