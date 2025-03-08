import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import zscore



###  Create two dataframe for Red and White

df_red=pd.read_csv("Dataset/wine_quality_Red.csv")
df_white=pd.read_csv("Dataset/wine_quality_White.csv")

# print(df_red.info())
# print(df_white.info())

#  basic information
# print(df_red.shape,df_white.shape)
# print(df_red.describe(),df_white.describe())

##  check data

# print(df_red.isnull().sum())
# print(df_white.isnull().sum())

# print(df_red.isna().sum())
# print(df_white.isna().sum())

# print(df_red.duplicated())
# print(df_white.duplicated())

df_red=df_red.drop_duplicates()
df_white=df_white.drop_duplicates()

# print(df_red.shape,df_white.shape)

##  Find Outliers - Zscore for all features

Zscore_red=zscore(df_red[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                        'pH', 'sulphates', 'alcohol']])
outliers_red=(abs(Zscore_red)>3).sum(axis=0)
# # print how much bias
# print(outliers_red)

outliers_redmask = abs(Zscore_red) > 3  
outliers_reddata = df_red[outliers_redmask.any(axis=1)]  
# # print details of bias
# print(outliers_reddata)


Zscore_white=zscore(df_white[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                        'pH', 'sulphates', 'alcohol']])
outliers_white=(abs(Zscore_white)>3).sum(axis=0)
# # print how much bias
# print(outliers_white)

outliers_whitemask = abs(Zscore_white) > 3  
outliers_whitedata = df_white[outliers_whitemask.any(axis=1)]  
# # print details of bias
# print(outliers_whitedata)

# delete all bias

df_red = df_red[(abs(Zscore_red) <= 3).all(axis=1)]
df_white = df_white[(abs(Zscore_white) <= 3).all(axis=1)]

print(df_red,df_white)







### Descriptive Analysis

# ##  Correlation with Pearson and Heatmap

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


# #  Histograms for Selected Variables

# Red_to_plot = ['citric acid', 'sulphates', 'alcohol','fixed acidity','residual sugar', 'quality']
# White_to_plot=['citric acid', 'sulphates', 'alcohol','free sulfur dioxide', 'pH','quality']

# # Layout Set and Histograms
# fig, axes=plt.subplots(2,6,figsize=(20, 10)) 


# # Red wine histogram 
# for i, column in enumerate(Red_to_plot):
#     ax = axes[0, i]
#     sb.histplot(df_red[column], kde=True, bins=10, color="#800020", edgecolor="black", ax=ax)
#     ax.set_title(f"Red_{column}", fontsize=12)
#     ax.set_xlabel('')

# # White wine histogram 
# for i, column in enumerate(White_to_plot):
#     ax = axes[1, i] 
#     sb.histplot(df_white[column], kde=True, bins=10, color="#800020", edgecolor="black", ax=ax)
#     ax.set_title(f"White_{column}", fontsize=12)
#     ax.set_xlabel('')

# plt.tight_layout()
# plt.show()







###  Model Building  

from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split, KFold

## train and test

xRed=df_red.drop('quality',axis=1)
yRed=df_red.drop('quality',axis=1)
xRed_train, xRed_test, yRed_train, yRed_test= train_test_split(xRed, yRed, test_size=0.2, random_state=42)

xWhite=df_white.drop('quality',axis=1)
yWhite=df_white.drop('quality',axis=1)
xWhite_train, xWhite_test, yWhite_train, yWhite_test= train_test_split(xWhite, yWhite, test_size=0.2, random_state=42)

## Decision Tree
from sklearn.tree import DecisionTreeRegressor

treeRed=DecisionTreeRegressor()
treeRed.fit(xRed_train,yRed_train)

preRed=treeRed.predict(xRed_test)
mse_treeRed = mean_squared_error(yRed_test, preRed)
mae_treeRed = mean_absolute_error(yRed_test, preRed)
r2_treeRed = r2_score(yRed_test, preRed)
print("According to the results obtained from the Decision Tree Regression model, the predicted values are as follows: ")
print("Mean Squared Error (MSE) :", mse_treeRed)






