import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Dataset/portwine.csv")

# Select relevant columns
df = df[[
    "wine_alcohol", "wine_price", "wine_point",
    "wine_country", "wine_variety", "wine_region", "year"
]]

# Fill missing string values
df["wine_country"].fillna("Unknown", inplace=True)
df["wine_variety"].fillna("Unknown", inplace=True)
df["wine_region"].fillna("Unknown", inplace=True)

# Fill missing values using country-level statistics
df["wine_alcohol"].fillna(df.groupby("wine_country")["wine_alcohol"].transform("mean"), inplace=True)
df["wine_alcohol"].fillna(df["wine_alcohol"].mean(), inplace=True)

df["wine_price"].fillna(df.groupby("wine_country")["wine_price"].transform("mean"), inplace=True)
df["wine_price"].fillna(df["wine_price"].mean(), inplace=True)

df["wine_point"].fillna(df.groupby("wine_country")["wine_point"].transform("mean"), inplace=True)
df["wine_point"].fillna(df["wine_point"].mean(), inplace=True)

df["year"].fillna(df.groupby("wine_country")["year"].transform("median"), inplace=True)
df["year"].fillna(df["year"].median(), inplace=True)

# Visualizations

# 1. Alcohol vs Price Scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='wine_alcohol', y='wine_price', hue='wine_country', alpha=0.6)
sns.regplot(data=df, x='wine_alcohol', y='wine_price', scatter=False, color='black')
plt.title("Alcohol Content vs Wine Price")
plt.xlabel("Alcohol (%)")
plt.ylabel("Wine Price ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Average Wine Score by Country
country_avg_points = df.groupby("wine_country")["wine_point"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
country_avg_points.plot(kind='bar')
plt.title("Average Wine Score by Country")
plt.ylabel("Average Score")
plt.xlabel("Country")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Alcohol vs Score by Country
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="wine_alcohol", y="wine_point", hue="wine_country", alpha=0.7)
plt.title("Alcohol vs. Score (by Country)")
plt.xlabel("Alcohol (%)")
plt.ylabel("Wine Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Price vs Score (under $300)
plt.figure(figsize=(10, 6))
sns.regplot(data=df[df["wine_price"] < 300], x="wine_price", y="wine_point", scatter_kws={'alpha':0.4})
plt.title("Wine Price vs. Score (Under $300)")
plt.xlabel("Wine Price ($)")
plt.ylabel("Wine Score")
plt.grid(True)
plt.tight_layout()
plt.show()
