import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import zscore
import time
import psutil
import os

from A_Preprocessing import (df_red,df_white)


###  Model Building  

from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

def get_memory_usage():
    """Return memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024) 

def print_model_results(model_name, wine_type, rmse, mae, r2, train_time, pred_time, memory_used):
    print(f"\n{'='*50}")
    print(f"{model_name} - {wine_type} Wine Results:")
    print(f"{'='*50}")
    print(f"Performance Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"Resource Usage:")
    print(f"  Training Time: {train_time:.4f} seconds")
    print(f"  Prediction Time: {pred_time:.4f} seconds")
    print(f"  Memory Used: {memory_used:.2f} MB")

## train and test

xRed=df_red[['citric acid', 'sulphates', 'alcohol','fixed acidity','residual sugar']]
yRed=df_red['quality']
xRed_train, xRed_test, yRed_train, yRed_test= train_test_split(xRed, yRed, test_size=0.2, random_state=42)

xWhite=df_white[['citric acid', 'sulphates', 'alcohol','free sulfur dioxide', 'pH']]
yWhite=df_white['quality']
xWhite_train, xWhite_test, yWhite_train, yWhite_test= train_test_split(xWhite, yWhite, test_size=0.2, random_state=42)


### Parameter grid 

## 1. DecisionTree
from sklearn.tree import DecisionTreeRegressor

dt_param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

## 2. RandomForest
from sklearn.ensemble import RandomForestRegressor
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
}

## 3. XGBoost
from xgboost import XGBRegressor
reduced_xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 8],
    'learning_rate': [0.05, 0.1],
}

## 4. LinearRegression
from sklearn.linear_model import LinearRegression

## 5. SVR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
svr_param_grid = {
    'C': [1, 5, 10],
    'epsilon': [0.1, 0.2, 0.5],
    'kernel': ['rbf', 'linear']
}



###### RED WINE MODELS ######

## 1. DecisionTree
dt_grid_red = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_param_grid, cv=5, 
                         scoring='neg_mean_squared_error', n_jobs=-1)
dt_grid_red.fit(xRed_train, yRed_train)
print(f"Best Decision Tree parameters for red wine: {dt_grid_red.best_params_}")

# Use best parameters
initial_memory = get_memory_usage()
start_time = time.time()
treeRed = DecisionTreeRegressor(**dt_grid_red.best_params_, random_state=42)
treeRed.fit(xRed_train, yRed_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
tree_preRed = treeRed.predict(xRed_test)
pred_time = time.time() - start_time

mse_treeRed = mean_squared_error(yRed_test, tree_preRed)
rmse_treeRed = np.sqrt(mse_treeRed)
mae_treeRed = mean_absolute_error(yRed_test, tree_preRed)
r2_treeRed = r2_score(yRed_test, tree_preRed)

print_model_results("Decision Tree", "Red", rmse_treeRed, mae_treeRed, r2_treeRed, 
                   train_time, pred_time, memory_used)




## 2. RandomForest
rf_grid_red = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, 
                          scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid_red.fit(xRed_train, yRed_train)
print(f"Best Random Forest parameters for red wine: {rf_grid_red.best_params_}")

# Use best parameters
initial_memory = get_memory_usage()
start_time = time.time()
RFRed = RandomForestRegressor(**rf_grid_red.best_params_, random_state=42)
RFRed.fit(xRed_train, yRed_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
RF_preRed = RFRed.predict(xRed_test)
pred_time = time.time() - start_time

mse_RFRed = mean_squared_error(yRed_test, RF_preRed)
rmse_RFRed = np.sqrt(mse_RFRed)
mae_RFRed = mean_absolute_error(yRed_test, RF_preRed)
r2_RFRed = r2_score(yRed_test, RF_preRed)

print_model_results("Random Forest", "Red", rmse_RFRed, mae_RFRed, r2_RFRed, 
                   train_time, pred_time, memory_used)

# Feature importance for Random Forest (Red wine)
print("\nRandom Forest Feature Importance (Red wine):")
rf_red_importance = dict(zip(xRed.columns, RFRed.feature_importances_))
sorted_rf_red_importance = {k: v for k, v in sorted(rf_red_importance.items(), key=lambda item: item[1], reverse=True)}
for feature, importance in sorted_rf_red_importance.items():
    print(f"{feature}: {importance:.4f}")




## 3. XGBoost
xgb_grid_red = GridSearchCV(XGBRegressor(random_state=42), reduced_xgb_param_grid, cv=5, 
                           scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid_red.fit(xRed_train, yRed_train)
print(f"Best XGBoost parameters for red wine: {xgb_grid_red.best_params_}")

# Use best parameters
initial_memory = get_memory_usage()
start_time = time.time()
XGBRed = XGBRegressor(**xgb_grid_red.best_params_, random_state=42)
XGBRed.fit(xRed_train, yRed_train, eval_set=[(xRed_test, yRed_test)], verbose=False)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
XGB_preRed = XGBRed.predict(xRed_test)
pred_time = time.time() - start_time

mse_XGBRed = mean_squared_error(yRed_test, XGB_preRed)
rmse_XGBRed = np.sqrt(mse_XGBRed)
mae_XGBRed = mean_absolute_error(yRed_test, XGB_preRed)
r2_XGBRed = r2_score(yRed_test, XGB_preRed)

print_model_results("XGBoost", "Red", rmse_XGBRed, mae_XGBRed, r2_XGBRed, 
                   train_time, pred_time, memory_used)




## 4. LinearRegression
initial_memory = get_memory_usage()
start_time = time.time()
LRALLRed = LinearRegression()
LRALLRed.fit(xRed_train, yRed_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
LRALL_preRed = LRALLRed.predict(xRed_test)
pred_time = time.time() - start_time

mse_LRALLRed = mean_squared_error(yRed_test, LRALL_preRed)
rmse_LRALLRed = np.sqrt(mse_LRALLRed)
mae_LRALLRed = mean_absolute_error(yRed_test, LRALL_preRed)
r2_LRALLRed = r2_score(yRed_test, LRALL_preRed)

print_model_results("Linear Regression", "Red", rmse_LRALLRed, mae_LRALLRed, r2_LRALLRed, 
                   train_time, pred_time, memory_used)

# Print coefficients
print("\nLinear Regression Coefficients (Red wine):")
lr_red_coeffs = dict(zip(xRed.columns, LRALLRed.coef_))
for feature, coef in lr_red_coeffs.items():
    print(f"{feature}: {coef:.4f}")



## 5. SVR
scaler_red = StandardScaler()
xRed_train_scaled = scaler_red.fit_transform(xRed_train)
xRed_test_scaled = scaler_red.transform(xRed_test)

svr_grid_red = GridSearchCV(SVR(), svr_param_grid, cv=5, 
                           scoring='neg_mean_squared_error', n_jobs=-1)
svr_grid_red.fit(xRed_train_scaled, yRed_train)
print(f"Best SVR parameters for red wine: {svr_grid_red.best_params_}")

# Use best parameters
initial_memory = get_memory_usage()
start_time = time.time()
SVRRed = SVR(**svr_grid_red.best_params_)
SVRRed.fit(xRed_train_scaled, yRed_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
SVR_preRed = SVRRed.predict(xRed_test_scaled)
pred_time = time.time() - start_time

mse_SVRRed = mean_squared_error(yRed_test, SVR_preRed)
rmse_SVRRed = np.sqrt(mse_SVRRed)
mae_SVRRed = mean_absolute_error(yRed_test, SVR_preRed)
r2_SVRRed = r2_score(yRed_test, SVR_preRed)

print_model_results("SVR", "Red", rmse_SVRRed, mae_SVRRed, r2_SVRRed, 
                   train_time, pred_time, memory_used)






###### WHITE WINE MODELS ######

## 1. DecisionTree
dt_grid_white = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_param_grid, cv=5, 
                            scoring='neg_mean_squared_error', n_jobs=-1)
dt_grid_white.fit(xWhite_train, yWhite_train)
print(f"Best Decision Tree parameters for white wine: {dt_grid_white.best_params_}")

# Use best parameters
initial_memory = get_memory_usage()
start_time = time.time()
treeWhite = DecisionTreeRegressor(**dt_grid_white.best_params_, random_state=42)
treeWhite.fit(xWhite_train, yWhite_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
tree_preWhite = treeWhite.predict(xWhite_test)
pred_time = time.time() - start_time

mse_treeWhite = mean_squared_error(yWhite_test, tree_preWhite)
rmse_treeWhite = np.sqrt(mse_treeWhite)
mae_treeWhite = mean_absolute_error(yWhite_test, tree_preWhite)
r2_treeWhite = r2_score(yWhite_test, tree_preWhite)

print_model_results("Decision Tree", "White", rmse_treeWhite, mae_treeWhite, r2_treeWhite, 
                   train_time, pred_time, memory_used)




## 2. RandomForest
rf_grid_white = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, 
                            scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid_white.fit(xWhite_train, yWhite_train)
print(f"Best Random Forest parameters for white wine: {rf_grid_white.best_params_}")

# Use best parameters
initial_memory = get_memory_usage()
start_time = time.time()
RFWhite = RandomForestRegressor(**rf_grid_white.best_params_, random_state=42)
RFWhite.fit(xWhite_train, yWhite_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
RF_preWhite = RFWhite.predict(xWhite_test)
pred_time = time.time() - start_time

mse_RFWhite = mean_squared_error(yWhite_test, RF_preWhite)
rmse_RFWhite = np.sqrt(mse_RFWhite)
mae_RFWhite = mean_absolute_error(yWhite_test, RF_preWhite)
r2_RFWhite = r2_score(yWhite_test, RF_preWhite)

print_model_results("Random Forest", "White", rmse_RFWhite, mae_RFWhite, r2_RFWhite, 
                   train_time, pred_time, memory_used)

# Feature importance for Random Forest (White wine)
print("\nRandom Forest Feature Importance (White wine):")
rf_white_importance = dict(zip(xWhite.columns, RFWhite.feature_importances_))
sorted_rf_white_importance = {k: v for k, v in sorted(rf_white_importance.items(), key=lambda item: item[1], reverse=True)}
for feature, importance in sorted_rf_white_importance.items():
    print(f"{feature}: {importance:.4f}")




## 3. XGBoost
xgb_grid_white = GridSearchCV(XGBRegressor(random_state=42), reduced_xgb_param_grid, cv=5, 
                             scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid_white.fit(xWhite_train, yWhite_train)
print(f"Best XGBoost parameters for white wine: {xgb_grid_white.best_params_}")

# Use best parameters
initial_memory = get_memory_usage()
start_time = time.time()
XGBWhite = XGBRegressor(**xgb_grid_white.best_params_, random_state=42)
XGBWhite.fit(xWhite_train, yWhite_train, eval_set=[(xWhite_test, yWhite_test)], verbose=False)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
XGB_preWhite = XGBWhite.predict(xWhite_test)
pred_time = time.time() - start_time

mse_XGBWhite = mean_squared_error(yWhite_test, XGB_preWhite)
rmse_XGBWhite = np.sqrt(mse_XGBWhite)
mae_XGBWhite = mean_absolute_error(yWhite_test, XGB_preWhite)
r2_XGBWhite = r2_score(yWhite_test, XGB_preWhite)

print_model_results("XGBoost", "White", rmse_XGBWhite, mae_XGBWhite, r2_XGBWhite, 
                   train_time, pred_time, memory_used)




## 4. LinearRegression
initial_memory = get_memory_usage()
start_time = time.time()
LRALLWhite = LinearRegression()
LRALLWhite.fit(xWhite_train, yWhite_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
LRALL_preWhite = LRALLWhite.predict(xWhite_test)
pred_time = time.time() - start_time

mse_LRALLWhite = mean_squared_error(yWhite_test, LRALL_preWhite)
rmse_LRALLWhite = np.sqrt(mse_LRALLWhite)
mae_LRALLWhite = mean_absolute_error(yWhite_test, LRALL_preWhite)
r2_LRALLWhite = r2_score(yWhite_test, LRALL_preWhite)

print_model_results("Linear Regression", "White", rmse_LRALLWhite, mae_LRALLWhite, r2_LRALLWhite, 
                   train_time, pred_time, memory_used)

# Print coefficients
print("\nLinear Regression Coefficients (White wine):")
lr_white_coeffs = dict(zip(xWhite.columns, LRALLWhite.coef_))
for feature, coef in lr_white_coeffs.items():
    print(f"{feature}: {coef:.4f}")



## 5. SVR
scaler_white = StandardScaler()
xWhite_train_scaled = scaler_white.fit_transform(xWhite_train)
xWhite_test_scaled = scaler_white.transform(xWhite_test)

# White wine SVR tuning
svr_grid_white = GridSearchCV(SVR(), svr_param_grid, cv=5, 
                             scoring='neg_mean_squared_error', n_jobs=-1)
svr_grid_white.fit(xWhite_train_scaled, yWhite_train)
print(f"Best SVR parameters for white wine: {svr_grid_white.best_params_}")

# Use best parameters
initial_memory = get_memory_usage()
start_time = time.time()
SVRWhite = SVR(**svr_grid_white.best_params_)
SVRWhite.fit(xWhite_train_scaled, yWhite_train)
train_time = time.time() - start_time
current_memory = get_memory_usage()
memory_used = current_memory - initial_memory

start_time = time.time()
SVR_preWhite = SVRWhite.predict(xWhite_test_scaled)
pred_time = time.time() - start_time

mse_SVRWhite = mean_squared_error(yWhite_test, SVR_preWhite)
rmse_SVRWhite = np.sqrt(mse_SVRWhite)
mae_SVRWhite = mean_absolute_error(yWhite_test, SVR_preWhite)
r2_SVRWhite = r2_score(yWhite_test, SVR_preWhite)

print_model_results("SVR", "White", rmse_SVRWhite, mae_SVRWhite, r2_SVRWhite, 
                   train_time, pred_time, memory_used)












