# Impor the necessary libraries
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

"""
Analysis of feature importance
"""

# Create a random dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)


start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time
forest_importances = pd.Series(importances, index=feature_names)

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")



fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

import seaborn as sns
import matplotlib.pyplot as plt

start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

# Set the style to 'whitegrid' for a ggplot2 like appearance
sns.set_style("whitegrid")

plt.figure(figsize=(10, 8))
sns.barplot(x=forest_importances, y=forest_importances.index, xerr=std, color='b', palette='muted')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.text(0.5, 1.02, "Black lines represent standard deviation", transform=plt.gca().transAxes, ha='center', fontsize=12)
plt.tight_layout()
plt.show()