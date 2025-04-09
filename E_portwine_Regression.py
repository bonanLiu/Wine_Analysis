import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR

# Load and clean the dataset
df = pd.read_csv("Dataset/portwine.csv")
df = df[['wine_description', 'wine_point', 'wine_variety']].dropna()

import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_description'] = df['wine_description'].apply(clean_text)


X = df['clean_description']
y = df['wine_point']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Define models to compare
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "SVR (Linear Kernel)": SVR(kernel='linear')
}


results = {}

for name, model in models.items():
    print(f"Training: {name}")
    model.fit(X_train_tfidf.toarray(), y_train)
    y_pred = model.predict(X_test_tfidf.toarray())
    mse = mean_squared_error(y_test, y_pred)
    results[name] = round(mse, 2)


print("\n Model Performance (lower MSE = better):")
for name, mse in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name:<25}: MSE = {mse}")


import matplotlib.pyplot as plt

# Visualize model comparison
plt.figure(figsize=(10, 6))
names = list(results.keys())
scores = list(results.values())
bars = plt.barh(names, scores, color='skyblue')
plt.xlabel("Mean Squared Error (MSE)")
plt.title(" Model Comparison - Wine Score Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Ridge Regression Tuning
ridge = Ridge()

param_grid = {
    'alpha': [0.01, 0.1, 1.0, 3.0, 5.0, 10.0, 30.0, 100.0]
}

grid = GridSearchCV(
    ridge,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1
)

grid.fit(X_train_tfidf.toarray(), y_train)

# Best model evaluation
best_alpha = grid.best_params_['alpha']
best_score = -grid.best_score_

print(f"\n Best alpha: {best_alpha}")
print(f"Best CV Mean Squared Error: {best_score:.2f}")

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_tfidf.toarray())
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE with best alpha: {test_mse:.2f}")



import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal', edgecolors='k')
plt.plot([80, 100], [80, 100], 'r--', label="Perfect Prediction")

plt.title("Actual vs Predicted Wine Scores (Ridge Regression)")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.xlim(80, 100)
plt.ylim(80, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# WordCloud
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(["wine", "port", "sherry", "flavors", "drink", "will", "it's", "its", "notes", "this", "a", "the"])

groups = {
    "All Wines": df['wine_description'].dropna(),
    "Port Wines (90+)": df[(df['wine_point'] >= 90) & (df['wine_variety'] == 'Port')]['wine_description'].dropna()
}

for group_name, text_series in groups.items():
    text = " ".join(text_series.tolist())
    wc = WordCloud(width=1000, height=500, background_color='white', stopwords=custom_stopwords).generate(text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"WordCloud - {group_name}")
    plt.show()