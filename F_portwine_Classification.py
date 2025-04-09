import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# Load and Preprocess Data
new_df = pd.read_csv("Dataset/portwine.csv")

df = new_df.dropna(subset=['wine_point', 'wine_price', 'wine_country', 'wine_variety', 'wine_description'])
df = df[df['wine_point'] >= 80]
df['quality_label'] = df['wine_point'].apply(lambda x: 1 if x >= 90 else 0)

# Clean wine descriptions
cleaned_descriptions = []
for text in df['wine_description']:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    cleaned_descriptions.append(text)
df['clean_description'] = cleaned_descriptions

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=100)
X = tfidf.fit_transform(df['clean_description']).toarray()
y = df['quality_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cross Validation for Multiple Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_results[name] = scores
    print(f"{name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Hyperparameter Tuning for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 
                    param_grid=xgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
best_xgb = grid.best_estimator_
best_xgb.fit(X_train, y_train)
xgb_preds = best_xgb.predict(X_test)


# Visualize Cross-Validated Accuracies
cv_means = {name: np.mean(score) for name, score in cv_results.items()}
plt.figure(figsize=(6, 4))
sns.barplot(x=list(cv_means.keys()), y=list(cv_means.values()))
plt.ylabel("Cross-Validated Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.close()

# Final Evaluation Results
print("\n===== Final Evaluation: XGBoost on Test Set =====")
print("Best Parameters:", grid.best_params_)
print("Test Accuracy:", accuracy_score(y_test, xgb_preds))
print("\nClassification Report:")
print(classification_report(y_test, xgb_preds))
