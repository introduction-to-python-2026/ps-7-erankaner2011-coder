import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Inspect data
print(df.head())
print(df.describe())

x = df["sepal length (cm)"]
y = df["petal length (cm)"]

plt.figure()
plt.hist(x, bins=20)
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.hist(y, bins=20)
plt.title("Petal Length Distribution")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.scatter(x, y)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length")
plt.show()

correlation = df.corr()

plt.figure()
plt.scatter(x, y)
plt.title(f"Correlation: {correlation.loc['sepal length (cm)', 'petal length (cm)']:.2f}")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")

# Save figure
plt.savefig("correlation_plot.png")
plt.show()

import numpy as np

m, b = np.polyfit(x, y, 1)

plt.figure()
plt.scatter(x, y)
plt.plot(x, m*x + b)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Scatter Plot with Regression Line")
plt.show()
