# Task 1: Load and explore the dataset

import pandas as pd
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Show first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Check data types
print(df.dtypes)

# Task 2: Basic Data Analysis

# Basic statistics
print(df.describe())

# Group by species
print(df.groupby("species").mean(numeric_only=True))

# Task 3: Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns

# Line chart (simulated time series)
df_sorted = df.sort_values(by="petal length (cm)").reset_index()
plt.plot(df_sorted.index, df_sorted['petal length (cm)'])
plt.title("Simulated Time Series of Petal Length")
plt.xlabel("Index")
plt.ylabel("Petal Length")
plt.show()

# Bar chart
df.groupby("species")["sepal width (cm)"].mean().plot(kind='bar')
plt.title("Average Sepal Width by Species")
plt.ylabel("Width (cm)")
plt.show()

# Histogram
plt.hist(df["petal length (cm)"], bins=20, color="skyblue", edgecolor="black")
plt.title("Petal Length Distribution")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

# Scatter plot
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df)
plt.title("Sepal Length vs Petal Length")
plt.show()

