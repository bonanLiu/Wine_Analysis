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

df_red=df_red.drop(columns=['quality'])
df_white=df_white.drop(columns=['quality'])

# print(df_red['quality_new'].value_counts())  #balance
# print(df_white['quality_new'].value_counts()) #inbalance
# print(df_red,df_white.shape)


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


### PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca_features=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
top_n = 5

## Red
xRed_pcaChosen=df_red[pca_features]
Red_scaler=StandardScaler()
xRed_scale=Red_scaler.fit_transform(xRed_pcaChosen)

pca_Red_Test=PCA(n_components=len(pca_features))
xRed_pcaTest=pca_Red_Test.fit_transform(xRed_scale)

top_Red_pcs = [f"PC{i+1}" for i in range(top_n)]
load_allRed_pca = pd.DataFrame(pca_Red_Test.components_.T,
                            index=pca_features,
                            columns=[f"PC{i+1}" for i in range(len(pca_features))])

feature_Red_importance = load_allRed_pca[top_Red_pcs].abs().sum(axis=1).sort_values(ascending=False)
# print("Top 5 Features of Red Wine:\n")
# print(feature_Red_importance.head(5))

Red_top5_feature=feature_Red_importance.head(5).index.tolist()
# print(Red_top5_feature)


## Wine
xWhite_pcaChosen=df_white[pca_features]
White_scaler=StandardScaler()
xWhite_scale=White_scaler.fit_transform(xWhite_pcaChosen)

pca_White_Test=PCA(n_components=len(pca_features))
xWhite_pcaTest=pca_White_Test.fit_transform(xWhite_scale)

top_White_pcs = [f"PC{i+1}" for i in range(top_n)]
load_allWhite_pca = pd.DataFrame(pca_White_Test.components_.T,
                            index=pca_features,
                            columns=[f"PC{i+1}" for i in range(len(pca_features))])

feature_White_importance = load_allWhite_pca[top_White_pcs].abs().sum(axis=1).sort_values(ascending=False)
# print("Top 5 Features of White Wine:\n")
# print(feature_White_importance.head(5))

White_top5_feature=feature_White_importance.head(5).index.tolist()



## train and test
from sklearn.model_selection import train_test_split

x_Red=df_red[Red_top5_feature]
y_Red=df_red['quality_new']
x_Red_train, x_Red_test, y_Red_train, y_Red_test= train_test_split(x_Red, y_Red, test_size=0.15, random_state=42,stratify=y_Red)
# print(x_Red,y_Red)

x_White=df_white[White_top5_feature]
y_White=df_white['quality_new']
x_White_train, x_White_test, y_White_train, y_White_test= train_test_split(x_White, y_White, test_size=0.2, random_state=42,stratify=y_White)
# print(x_White,y_White)





## RandomForest - Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Red
RFC_Red=RandomForestClassifier(n_estimators=50, max_depth=7, class_weight='balanced', random_state=0)
RFC_Red.fit(x_Red_train,y_Red_train)
RFC_Red_pre=RFC_Red.predict(x_Red_test)


accuracy_Red_RFC=accuracy_score(y_Red_test, RFC_Red_pre)
# print(f"Random Forest Accuracy in Red Wine: {accuracy_Red_RFC:.4f}")
# print(classification_report(y_Red_test, RFC_Red_pre))

# White
RFC_White=RandomForestClassifier(n_estimators=30, max_depth=7, random_state=0)
RFC_White.fit(x_White_train,y_White_train)
RFC_White_pre=RFC_White.predict(x_White_test)

accuracy_White_RFC=accuracy_score(y_White_test, RFC_White_pre)
# print(f"Random Forest Accuracy in White Wine: {accuracy_White_RFC:.4f}")
# print(classification_report(y_White_test, RFC_White_pre))


print("\n")

## XGBoost - Classifier
from xgboost import XGBClassifier
XGB_Red=XGBClassifier(n_estimators=120, learning_rate=0.0655, max_depth=6, reg_alpha=0.02, random_state=42)
XGB_Red.fit(x_Red_train,y_Red_train)
XGB_Red_pre=XGB_Red.predict(x_Red_test)

accuracy_Red_XGB=accuracy_score(y_Red_test, XGB_Red_pre)
print(f"XGBoost Accuracy in Red Wine: {accuracy_Red_XGB:.4f}")
# print(classification_report(y_Red_test, XGB_Red_pre))

# White
XGB_White=XGBClassifier(n_estimators=70, learning_rate=0.075, max_depth=6, reg_alpha=0.01, random_state=0,scale_pos_weight=2.23)
XGB_White.fit(x_White_train,y_White_train)
XGB_White_pre=XGB_White.predict(x_White_test)

accuracy_White_XGB=accuracy_score(y_White_test, XGB_White_pre)
print(f"XGBoost Accuracy in White Wine: {accuracy_White_XGB:.4f}")
# print(classification_report(y_White_test, XGB_White_pre))



## SVM




