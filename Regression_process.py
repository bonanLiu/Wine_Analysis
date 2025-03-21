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

# print(df_red.shape,df_white.shape)







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
from sklearn.model_selection import cross_val_score, train_test_split

## train and test

xRed=df_red[['citric acid', 'sulphates', 'alcohol','fixed acidity','residual sugar']]
yRed=df_red['quality']
xRed_train, xRed_test, yRed_train, yRed_test= train_test_split(xRed, yRed, test_size=0.2, random_state=42)

xWhite=df_white[['citric acid', 'sulphates', 'alcohol','free sulfur dioxide', 'pH']]
yWhite=df_white['quality']
xWhite_train, xWhite_test, yWhite_train, yWhite_test= train_test_split(xWhite, yWhite, test_size=0.2, random_state=42)

## DecisionTree
from sklearn.tree import DecisionTreeRegressor


treeRed=DecisionTreeRegressor()
treeRed.fit(xRed_train,yRed_train)
tree_preRed = treeRed.predict(xRed_test)
mse_treeRed = mean_squared_error(yRed_test, tree_preRed)
rmse_treeRed = np.sqrt(mse_treeRed)
mae_treeRed = mean_absolute_error(yRed_test, tree_preRed)
r2_treeRed = r2_score(yRed_test, tree_preRed)

treeWhite=DecisionTreeRegressor()
treeWhite.fit(xWhite_train,yWhite_train)
tree_preWhite = treeWhite.predict(xWhite_test)
mse_treeWhite = mean_squared_error(yWhite_test, tree_preWhite)
rmse_treeWhite = np.sqrt(mse_treeWhite)
mae_treeWhite = mean_absolute_error(yWhite_test, tree_preWhite)
r2_treeWhite = r2_score(yWhite_test, tree_preWhite)




## RandomForest
from sklearn.ensemble import RandomForestRegressor

RFRed=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
RFRed.fit(xRed_train,yRed_train)
RF_preRed = RFRed.predict(xRed_test)
mse_RFRed = mean_squared_error(yRed_test, RF_preRed)
rmse_RFRed = np.sqrt(mse_RFRed)
mae_RFRed = mean_absolute_error(yRed_test, RF_preRed)
r2_RFRed = r2_score(yRed_test, RF_preRed)

RFWhite=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
RFWhite.fit(xWhite_train,yWhite_train)
RF_preWhite = RFWhite.predict(xWhite_test)
mse_RFWhite = mean_squared_error(yWhite_test, RF_preWhite)
rmse_RFWhite = np.sqrt(mse_RFWhite)
mae_RFWhite = mean_absolute_error(yWhite_test, RF_preWhite)
r2_RFWhite = r2_score(yWhite_test, RF_preWhite)







## XGBoost
from xgboost import XGBRegressor
XGBRed=XGBRegressor(n_estimators=100,max_depth=8,learning_rate=0.05,random_state=42)
XGBRed.fit(xRed_train,yRed_train,eval_set=[(xRed_test,yRed_test)],verbose=False)
XGB_preRed = XGBRed.predict(xRed_test)
mse_XGBRed = mean_squared_error(yRed_test, XGB_preRed)
rmse_XGBRed = np.sqrt(mse_XGBRed)
mae_XGBRed = mean_absolute_error(yRed_test, XGB_preRed)
r2_XGBRed = r2_score(yRed_test, XGB_preRed)

XGBWhite=XGBRegressor(n_estimators=100,max_depth=8,learning_rate=0.05,random_state=42)
XGBWhite.fit(xWhite_train,yWhite_train,eval_set=[(xWhite_test,yWhite_test)],verbose=False)
XGB_preWhite = XGBWhite.predict(xWhite_test)
mse_XGBWhite = mean_squared_error(yWhite_test, XGB_preWhite)
rmse_XGBWhite = np.sqrt(mse_XGBWhite)
mae_XGBWhite = mean_absolute_error(yWhite_test, XGB_preWhite)
r2_XGBWhite = r2_score(yWhite_test, XGB_preWhite)





## LinearRegression
from sklearn.linear_model import LinearRegression
LRALLRed=LinearRegression()
LRALLRed.fit(xRed_train,yRed_train)
LRALL_preRed = LRALLRed.predict(xRed_test)
mse_LRALLRed = mean_squared_error(yRed_test, LRALL_preRed)
rmse_LRALLRed = np.sqrt(mse_LRALLRed)
mae_LRALLRed = mean_absolute_error(yRed_test, LRALL_preRed)
r2_LRALLRed = r2_score(yRed_test, LRALL_preRed)

LRALLWhite=LinearRegression()
LRALLWhite.fit(xWhite_train,yWhite_train)
LRALL_preWhite = LRALLWhite.predict(xWhite_test)
mse_LRALLWhite = mean_squared_error(yWhite_test, LRALL_preWhite)
rmse_LRALLWhite = np.sqrt(mse_LRALLWhite)
mae_LRALLWhite = mean_absolute_error(yWhite_test, LRALL_preWhite)
r2_LRALLWhite = r2_score(yWhite_test, LRALL_preWhite)



## SVR
from sklearn.svm import SVR
SVRRed=SVR(kernel='rbf', C=7.0, epsilon=0.1)
SVRRed.fit(xRed_train,yRed_train)
SVR_preRed = SVRRed.predict(xRed_test)
mse_SVRRed = mean_squared_error(yRed_test, SVR_preRed)
rmse_SVRRed = np.sqrt(mse_SVRRed)
mae_SVRRed = mean_absolute_error(yRed_test, SVR_preRed)
r2_SVRRed = r2_score(yRed_test, SVR_preRed)

SVRWhite=SVR(kernel='rbf', C=7.0, epsilon=0.1)
SVRWhite.fit(xWhite_train,yWhite_train)
SVR_preWhite = SVRWhite.predict(xWhite_test)
mse_SVRWhite = mean_squared_error(yWhite_test, SVR_preWhite)
rmse_SVRWhite = np.sqrt(mse_SVRWhite)
mae_SVRWhite = mean_absolute_error(yWhite_test, SVR_preWhite)
r2_SVRWhite = r2_score(yWhite_test, SVR_preWhite)




RedWine_df = pd.DataFrame({
    "MSE": [mse_treeRed, mse_RFRed, mse_XGBRed, mse_LRALLRed, mse_SVRRed],
    "RMSE": [rmse_treeRed, rmse_RFRed, rmse_XGBRed, rmse_LRALLRed, rmse_SVRRed],
    "MAE": [mae_treeRed, mae_RFRed, mae_XGBRed, mae_LRALLRed, mae_SVRRed],
    "R2": [r2_treeRed, r2_RFRed, r2_XGBRed, r2_LRALLRed, r2_SVRRed]
}, index=["Decision Tree", "Random Forest", "XGBoost", "Linear Regression", "SVR"])


WhiteWine_df = pd.DataFrame({
    "MSE": [mse_treeWhite, mse_RFWhite, mse_XGBWhite, mse_LRALLWhite, mse_SVRWhite],
    "RMSE": [rmse_treeWhite, rmse_RFWhite, rmse_XGBWhite, rmse_LRALLWhite, rmse_SVRWhite],
    "MAE": [mae_treeWhite, mae_RFWhite, mae_XGBWhite, mae_LRALLWhite, mae_SVRWhite],
    "R2": [r2_treeWhite, r2_RFWhite, r2_XGBWhite, r2_LRALLWhite, r2_SVRWhite]
}, index=["Decision Tree", "Random Forest", "XGBoost", "Linear Regression", "SVR"])

# print("Red Wine Model Performance:")
# print(RedWine_df)
# print("\nWhite Wine Model Performance:")
# print(WhiteWine_df)














# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# mse_treeRed_k = -cross_val_score(treeRed, xRed, yRed, cv=kf, scoring='neg_mean_squared_error').mean()
# r2_treeRed_k = cross_val_score(treeRed, xRed, yRed, cv=kf, scoring='r2').mean()