import pandas as pd
import numpy as np
import time
import psutil
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Import the processed dataframe from Regression.py
from A_Preprocessing import (df_red,df_white)

df_red['quality_new'] = np.where(df_red['quality'].between(1, 5), 0, 1)
df_white['quality_new'] = np.where(df_white['quality'].between(1, 5), 0, 1)

df_red=df_red.drop(columns=['quality'])
df_white=df_white.drop(columns=['quality'])




### PCA
from sklearn.decomposition import PCA

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
print("Top 5 Features of Red Wine:\n")
print(feature_Red_importance.head(5))

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
print("Top 5 Features of White Wine:\n")
print(feature_White_importance.head(5))

White_top5_feature=feature_White_importance.head(5).index.tolist()
print("\n\n\n")


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
print(f"\nRed wine class distribution - Training: {dict(pd.Series(y_Red_train).value_counts())}")
print(f"White wine class distribution - Training: {dict(pd.Series(y_White_train).value_counts())}")
print("\n\n\n")




def get_memory_usage():
    """Return memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024) 

def print_model_results(model_name, wine_type, accuracy, train_time, pred_time, memory_used=None):
    print(f"\n{'='*50}")
    print(f"{model_name} - {wine_type} Wine Classification Results:")
    print(f"{'='*50}")
    print(f"Performance Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"Resource Usage:")
    print(f"  Training Time: {train_time:.4f} seconds")
    print(f"  Prediction Time: {pred_time:.4f} seconds")
    if memory_used is not None:
        print(f"  Memory Used: {memory_used:.2f} MB")



## 1. Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 7, 9],
    'class_weight': [None, 'balanced'],
    'min_samples_split': [2, 5]
}

## 2. XGBoost
from xgboost import XGBClassifier
xgb_param_grid = {
    'n_estimators': [100, 120, 150],
    'learning_rate': [0.05, 0.065, 0.08],
    'max_depth': [5, 6, 7],
    'reg_alpha': [0.01, 0.02, 0.05]
}

xgb_param_grid_white = {
    'n_estimators': [70, 100, 150],
    'learning_rate': [0.05, 0.075, 0.1],
    'max_depth': [5, 6, 7],
    'reg_alpha': [0.01, 0.02, 0.05],
    'scale_pos_weight': [1, 2, 2.23]  # set weight
}

## 3. SVM
from sklearn.svm import SVC
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

## 4. Naive Bayes
from sklearn.naive_bayes import GaussianNB




###### RED WINE ######

## 1. Random Forest
rf_grid_red = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1)
rf_grid_red.fit(x_Red_train, y_Red_train)
print(f"Best Random Forest parameters for red wine: {rf_grid_red.best_params_}")


initial_memory = get_memory_usage()
start_time = time.time()
RFC_Red = RandomForestClassifier(**rf_grid_red.best_params_, random_state=42)
RFC_Red.fit(x_Red_train, y_Red_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
RFC_Red_pre = RFC_Red.predict(x_Red_test)
pred_time = time.time() - start_time

accuracy_Red_RFC = accuracy_score(y_Red_test, RFC_Red_pre)
print_model_results("Random Forest", "Red", accuracy_Red_RFC, train_time, pred_time, memory_used)


print("\nClassification Report:")
print(classification_report(y_Red_test, RFC_Red_pre))

print("\nConfusion Matrix:")
print(confusion_matrix(y_Red_test, RFC_Red_pre))

print("\nRandom Forest Feature Importance (Red wine):")
rf_red_importance = dict(zip(x_Red.columns, RFC_Red.feature_importances_))
sorted_rf_red_importance = {k: v for k, v in sorted(rf_red_importance.items(), key=lambda item: item[1], reverse=True)}
for feature, importance in sorted_rf_red_importance.items():
    print(f"{feature}: {importance:.4f}")



## 2. XGBoost - Classifier
xgb_grid_red = GridSearchCV(XGBClassifier(random_state=42), xgb_param_grid, cv=5, 
                           scoring='accuracy', n_jobs=-1)
xgb_grid_red.fit(x_Red_train, y_Red_train)
print(f"Best XGBoost parameters for red wine: {xgb_grid_red.best_params_}")

initial_memory = get_memory_usage()
start_time = time.time()
XGB_Red = XGBClassifier(**xgb_grid_red.best_params_, random_state=42)
XGB_Red.fit(x_Red_train, y_Red_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
XGB_Red_pre = XGB_Red.predict(x_Red_test)
pred_time = time.time() - start_time

accuracy_Red_XGB = accuracy_score(y_Red_test, XGB_Red_pre)
print_model_results("XGBoost", "Red", accuracy_Red_XGB, train_time, pred_time, memory_used)

print("\nClassification Report:")
print(classification_report(y_Red_test, XGB_Red_pre))

print("\nConfusion Matrix:")
print(confusion_matrix(y_Red_test, XGB_Red_pre))



## 3. SVM
scaler_red = StandardScaler()
x_Red_train_scaled = scaler_red.fit_transform(x_Red_train)
x_Red_test_scaled = scaler_red.transform(x_Red_test)

svm_grid_red = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1)
svm_grid_red.fit(x_Red_train_scaled, y_Red_train)
print(f"Best SVM parameters for red wine: {svm_grid_red.best_params_}")

initial_memory = get_memory_usage()
start_time = time.time()
SVM_Red = SVC(**svm_grid_red.best_params_, random_state=42, probability=True)
SVM_Red.fit(x_Red_train_scaled, y_Red_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
SVM_Red_pre = SVM_Red.predict(x_Red_test_scaled)
pred_time = time.time() - start_time

accuracy_Red_SVM = accuracy_score(y_Red_test, SVM_Red_pre)
print_model_results("SVM", "Red", accuracy_Red_SVM, train_time, pred_time, memory_used)

print("\nClassification Report:")
print(classification_report(y_Red_test, SVM_Red_pre))

print("\nConfusion Matrix:")
print(confusion_matrix(y_Red_test, SVM_Red_pre))


## 4. Naive Bayes
initial_memory = get_memory_usage()
start_time = time.time()
NB_Red = GaussianNB()
NB_Red.fit(x_Red_train_scaled, y_Red_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
NB_Red_pre = NB_Red.predict(x_Red_test_scaled)
pred_time = time.time() - start_time

accuracy_Red_NB = accuracy_score(y_Red_test, NB_Red_pre)
print_model_results("Naive Bayes", "Red", accuracy_Red_NB, train_time, pred_time, memory_used)

print("\nClassification Report:")
print(classification_report(y_Red_test, NB_Red_pre))

print("\nConfusion Matrix:")
print(confusion_matrix(y_Red_test, NB_Red_pre))




###### WHITE WINE ######

## 1. Random Forest
rf_grid_white = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, 
                           scoring='accuracy', n_jobs=-1)
rf_grid_white.fit(x_White_train, y_White_train)
print(f"Best Random Forest parameters for white wine: {rf_grid_white.best_params_}")

initial_memory = get_memory_usage()
start_time = time.time()
RFC_White = RandomForestClassifier(**rf_grid_white.best_params_, random_state=42)
RFC_White.fit(x_White_train, y_White_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
RFC_White_pre = RFC_White.predict(x_White_test)
pred_time = time.time() - start_time

accuracy_White_RFC = accuracy_score(y_White_test, RFC_White_pre)
print_model_results("Random Forest", "White", accuracy_White_RFC, train_time, pred_time, memory_used)

print("\nClassification Report:")
print(classification_report(y_White_test, RFC_White_pre))

print("\nConfusion Matrix:")
print(confusion_matrix(y_White_test, RFC_White_pre))

print("\nRandom Forest Feature Importance (White wine):")
rf_white_importance = dict(zip(x_White.columns, RFC_White.feature_importances_))
sorted_rf_white_importance = {k: v for k, v in sorted(rf_white_importance.items(), key=lambda item: item[1], reverse=True)}
for feature, importance in sorted_rf_white_importance.items():
    print(f"{feature}: {importance:.4f}")



## 2. XGBoost - Classifier
xgb_grid_white = GridSearchCV(XGBClassifier(random_state=42), xgb_param_grid_white, cv=5, 
                            scoring='accuracy', n_jobs=-1)
xgb_grid_white.fit(x_White_train, y_White_train)
print(f"Best XGBoost parameters for white wine: {xgb_grid_white.best_params_}")

initial_memory = get_memory_usage()
start_time = time.time()
XGB_White = XGBClassifier(**xgb_grid_white.best_params_, random_state=42)
XGB_White.fit(x_White_train, y_White_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
XGB_White_pre = XGB_White.predict(x_White_test)
pred_time = time.time() - start_time

accuracy_White_XGB = accuracy_score(y_White_test, XGB_White_pre)
print_model_results("XGBoost", "White", accuracy_White_XGB, train_time, pred_time, memory_used)

print("\nClassification Report:")
print(classification_report(y_White_test, XGB_White_pre))

print("\nConfusion Matrix:")
print(confusion_matrix(y_White_test, XGB_White_pre))



## 3. SVM
scaler_white = StandardScaler()
x_White_train_scaled = scaler_white.fit_transform(x_White_train)
x_White_test_scaled = scaler_white.transform(x_White_test)

svm_grid_white = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, 
                           scoring='accuracy', n_jobs=-1)
svm_grid_white.fit(x_White_train_scaled, y_White_train)
print(f"Best SVM parameters for white wine: {svm_grid_white.best_params_}")

initial_memory = get_memory_usage()
start_time = time.time()
SVM_White = SVC(**svm_grid_white.best_params_, random_state=42, probability=True)
SVM_White.fit(x_White_train_scaled, y_White_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
SVM_White_pre = SVM_White.predict(x_White_test_scaled)
pred_time = time.time() - start_time

accuracy_White_SVM = accuracy_score(y_White_test, SVM_White_pre)
print_model_results("SVM", "White", accuracy_White_SVM, train_time, pred_time, memory_used)

print("\nClassification Report:")
print(classification_report(y_White_test, SVM_White_pre))

print("\nConfusion Matrix:")
print(confusion_matrix(y_White_test, SVM_White_pre))


## 4. Naive Bayes
initial_memory = get_memory_usage()
start_time = time.time()
NB_White = GaussianNB()
NB_White.fit(x_White_train_scaled, y_White_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
NB_White_pre = NB_White.predict(x_White_test_scaled)
pred_time = time.time() - start_time

accuracy_White_NB = accuracy_score(y_White_test, NB_White_pre)
print_model_results("Naive Bayes", "White", accuracy_White_NB, train_time, pred_time, memory_used)

print("\nClassification Report:")
print(classification_report(y_White_test, NB_White_pre))

print("\nConfusion Matrix:")
print(confusion_matrix(y_White_test, NB_White_pre))


