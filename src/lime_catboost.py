import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

# Load the German Credit dataset
data = pd.read_csv(r'../data/raw/german_processed.csv')

# Let's assume that 'Risk' is the target variable
y = data['GoodCustomer']
X = data.drop(['Gender', 'PurposeOfLoan'] , axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a CatBoostClassifier
clf = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, loss_function='Logloss')
clf.fit(X_train, y_train)

# Create a LimeTabularExplainer
explainer = LimeTabularExplainer(X_train.values, 
                                 feature_names=X_train.columns, 
                                 class_names=['Good', 'Bad'], 
                                 verbose=True, 
                                 mode='classification')

# Explain a prediction
i = np.random.randint(0, X_test.shape[0])
exp = explainer.explain_instance(X_test.values[i], clf.predict_proba, num_features=5)

# Print the explanation
print(exp.as_list())

from collections import defaultdict

# Initialize a dictionary to store the sum of LIME values for each feature
lime_sums = defaultdict(int)
# Initialize a dictionary to store the count of LIME values for each feature
lime_counts = defaultdict(int)

# Loop over all instances in the dataset
for i in range(X.shape[0]):
    # Explain the prediction for the current instance
    exp = explainer.explain_instance(X.values[i], clf.predict_proba, num_features=len(X.columns))
    
    # Loop over all features and their corresponding LIME values
    for feature, value in exp.as_list():
        # Add the absolute value of the LIME value to the sum for the current feature
        lime_sums[feature] += abs(value)
        # Increment the count of LIME values for the current feature
        lime_counts[feature] += 1

# Calculate the mean absolute LIME value for each feature
lime_means = {feature: lime_sums[feature] / lime_counts[feature] for feature in lime_sums}

# Convert the results to a list of tuples and print it
lime_means_list = list(lime_means.items())
print(lime_means_list)